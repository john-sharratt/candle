//! The screens' windows (§8): one borderless window per output, each
//! drawing a MilkDrop scene with lactea from the room's captured audio and
//! the beat, and ZenJ's own layer over it with glyphon. One GPU device for
//! all of them; every window redraws each frame the display gives it.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwap;
use chrono::{DateTime, Local};
use glyphon::cosmic_text::Align;
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use lactea::{Engine, EngineConfig, FrameClock, Transition};
use rtrb::Consumer;
use wgpu::{
    Adapter, Color as Fill, CommandEncoderDescriptor, CompositeAlphaMode, CurrentSurfaceTexture, Device, DeviceDescriptor,
    Instance, InstanceDescriptor, LoadOp, MultisampleState, Operations, PresentMode, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions, StoreOp, Surface,
    SurfaceColorSpace, SurfaceConfiguration, TextureFormat, TextureUsages, TextureViewDescriptor,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::{Fullscreen, Window, WindowId};

use super::overlay::{overlay, Overlay};
use super::scene::{presets, Scenes};
use crate::config::ScreensCfg;
use crate::live::Live;

/// What every window reads.
pub struct Shared {
    pub live: Arc<ArcSwap<Live>>,
    pub midnight: DateTime<Local>,
    pub sample_rate: u32,
}

struct Gpu {
    instance: Instance,
    device: Device,
    queue: Queue,
    adapter: Adapter,
}

struct Screen {
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    engine: Option<Engine>,
    scenes: Scenes,
    latency_ms: f64,
    pcm: Consumer<f32>,
    pcm_buf: Vec<f32>,
    atlas: TextAtlas,
    text: TextRenderer,
    viewport: Viewport,
    corner: Buffer,
    big: Buffer,
    frame: u32,
    // Dropped last: the surface borrows it.
    window: Arc<Window>,
}

/// Load `scenes`' current preset from `library` into `engine`; one that does
/// not load is passed over, a few at most.
fn load(library: &[PathBuf], engine: &mut Engine, scenes: &mut Scenes, transition: Transition) {
    for _ in 0..library.len().min(8) {
        let Some(i) = scenes.current() else { return };
        let path = &library[i];
        let name = path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let loaded = std::fs::read(path)
            .map_err(|e| e.to_string())
            .and_then(|b| engine.load_preset(&b, &name, transition).map(|_| ()).map_err(|e| e.to_string()));
        match loaded {
            Ok(()) => return,
            Err(e) => {
                tracing::debug!("screens: {name}: {e}");
                scenes.skip();
            }
        }
    }
}

pub struct Screens {
    cfg: ScreensCfg,
    shared: Shared,
    pcm: Vec<Option<Consumer<f32>>>,
    gpu: Option<Gpu>,
    screens: Vec<(WindowId, Screen)>,
    fonts: FontSystem,
    swash: SwashCache,
    library: Vec<PathBuf>,
    start: Instant,
}

impl Screens {
    /// The screens `cfg` names, each fed by its own of `pcm`.
    pub fn new(cfg: ScreensCfg, shared: Shared, pcm: Vec<Consumer<f32>>) -> Screens {
        let library = presets(&cfg.presets);
        tracing::info!(presets = library.len(), "screens: scene library read");
        Screens {
            cfg,
            shared,
            pcm: pcm.into_iter().map(Some).collect(),
            gpu: None,
            screens: Vec::new(),
            fonts: FontSystem::new(),
            swash: SwashCache::new(),
            library,
            start: Instant::now(),
        }
    }

    fn open(&mut self, el: &ActiveEventLoop) {
        let monitors: Vec<_> = el.available_monitors().collect();
        let total = self.cfg.outputs.len();
        for (n, out) in self.cfg.outputs.clone().into_iter().enumerate() {
            let Some(monitor) = monitors.get(out.monitor).cloned() else {
                tracing::warn!(monitor = out.monitor, found = monitors.len(), "screens: no such monitor");
                continue;
            };
            let attrs = Window::default_attributes()
                .with_title("ZenJ")
                .with_fullscreen(Some(Fullscreen::Borderless(Some(monitor))));
            let window = match el.create_window(attrs) {
                Ok(w) => Arc::new(w),
                Err(e) => {
                    tracing::warn!("screens: a window could not open: {e}");
                    continue;
                }
            };
            if self.gpu.is_none() {
                let instance = Instance::new(InstanceDescriptor::new_with_display_handle(Box::new(
                    el.owned_display_handle(),
                )));
                let adapter = match pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default())) {
                    Ok(a) => a,
                    Err(e) => {
                        tracing::warn!("screens: no GPU adapter: {e}");
                        return;
                    }
                };
                let (device, queue) = match pollster::block_on(adapter.request_device(&DeviceDescriptor::default())) {
                    Ok(dq) => dq,
                    Err(e) => {
                        tracing::warn!("screens: no GPU device: {e}");
                        return;
                    }
                };
                self.gpu = Some(Gpu { instance, device, queue, adapter });
            }
            let gpu = self.gpu.as_ref().expect("made above");
            let surface = match gpu.instance.create_surface(window.clone()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("screens: no surface: {e}");
                    continue;
                }
            };
            let caps = surface.get_capabilities(&gpu.adapter);
            let format = caps
                .formats
                .iter()
                .copied()
                .find(|f| *f == TextureFormat::Bgra8UnormSrgb)
                .or_else(|| caps.formats.first().copied())
                .unwrap_or(TextureFormat::Bgra8UnormSrgb);
            let size = window.inner_size();
            let config = SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: PresentMode::Fifo,
                alpha_mode: CompositeAlphaMode::Opaque,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: SurfaceColorSpace::Auto,
            };
            surface.configure(&gpu.device, &config);
            let mut scenes = Scenes::new(self.library.len(), n, total);
            let mut engine = Engine::new(
                &gpu.device,
                &gpu.queue,
                EngineConfig {
                    width: config.width,
                    height: config.height,
                    output_format: format,
                    sample_rate: self.shared.sample_rate,
                    seed: n as u64 + 1,
                    ..EngineConfig::default()
                },
            )
            .map_err(|e| tracing::warn!("screens: the scene engine could not start: {e}"))
            .ok();
            if let Some(e) = engine.as_mut() {
                load(&self.library, e, &mut scenes, Transition::Cut);
            }
            let cache = Cache::new(&gpu.device);
            let viewport = Viewport::new(&gpu.device, &cache);
            let mut atlas = TextAtlas::new(&gpu.device, &gpu.queue, &cache, format);
            let text = TextRenderer::new(&mut atlas, &gpu.device, MultisampleState::default(), None);
            let mut corner = Buffer::new(&mut self.fonts, Metrics::new(34.0, 44.0));
            corner.set_size(Some(config.width as f32 - 80.0), Some(config.height as f32));
            let mut big = Buffer::new(&mut self.fonts, Metrics::new(320.0, 340.0));
            big.set_size(Some(config.width as f32), Some(config.height as f32));
            let pcm = self.pcm.get_mut(n).and_then(Option::take);
            let Some(pcm) = pcm else {
                tracing::warn!(screen = n, "screens: no audio for this screen");
                continue;
            };
            self.screens.push((
                window.id(),
                Screen {
                    surface,
                    config,
                    engine,
                    scenes,
                    latency_ms: out.latency_ms,
                    pcm,
                    pcm_buf: Vec::new(),
                    atlas,
                    text,
                    viewport,
                    corner,
                    big,
                    frame: 0,
                    window,
                },
            ));
        }
        tracing::info!(open = self.screens.len(), "screens: open");
    }

    fn draw(&mut self, id: WindowId) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        let Some(i) = self.screens.iter().position(|(w, _)| *w == id) else { return };
        let live = self.shared.live.load_full();
        let now = Local::now();
        let ov = overlay(&live, now, self.shared.midnight);
        let air = live.decks.iter().find(|d| d.on_air);
        let time = self.start.elapsed().as_secs_f64();
        let library = &self.library;
        let s = &mut self.screens[i].1;

        // The scene: fed, changed on a phrase, drawn.
        s.pcm_buf.clear();
        while let Ok(x) = s.pcm.pop() {
            s.pcm_buf.push(x);
        }
        if let Some(e) = s.engine.as_mut() {
            if !s.pcm_buf.is_empty() {
                e.push_pcm_f32(&s.pcm_buf, 1);
            }
        }
        let change = air
            .and_then(|d| d.beat.map(|b| (b, d.bpm)))
            .and_then(|(b, bpm)| s.scenes.at_beat(b, bpm, s.latency_ms));
        if change.is_some() {
            if let Some(e) = s.engine.as_mut() {
                load(library, e, &mut s.scenes, Transition::Soft { seconds: 2.0 });
            }
        }

        // ZenJ's layer.
        let (w, h) = (s.config.width as f32, s.config.height as f32);
        let attrs = Attrs::new().family(Family::SansSerif);
        let area = match &ov {
            Overlay::Corner(lines) => {
                s.corner.set_text(&lines.join("\n"), &attrs, Shaping::Advanced, None);
                s.corner.shape_until_scroll(&mut self.fonts, false);
                let top = h - 44.0 * lines.len() as f32 - 48.0;
                (true, 40.0, top)
            }
            Overlay::Big(text) => {
                s.big.set_text(text, &attrs, Shaping::Advanced, Some(Align::Center));
                s.big.shape_until_scroll(&mut self.fonts, false);
                (false, 0.0, h / 2.0 - 170.0)
            }
        };
        s.viewport.update(&gpu.queue, Resolution { width: s.config.width, height: s.config.height });
        let buffer = if area.0 { &s.corner } else { &s.big };
        if let Err(e) = s.text.prepare(
            &gpu.device,
            &gpu.queue,
            &mut self.fonts,
            &mut s.atlas,
            &s.viewport,
            [TextArea {
                buffer,
                left: area.1,
                top: area.2,
                scale: 1.0,
                bounds: TextBounds { left: 0, top: 0, right: w as i32, bottom: h as i32 },
                default_color: Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            }],
            &mut self.swash,
        ) {
            tracing::debug!("screens: text: {e}");
        }

        let frame = match s.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(f) => f,
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Suboptimal(_) => {
                s.surface.configure(&gpu.device, &s.config);
                return;
            }
            CurrentSurfaceTexture::Lost => {
                if let Ok(surface) = gpu.instance.create_surface(s.window.clone()) {
                    s.surface = surface;
                    s.surface.configure(&gpu.device, &s.config);
                }
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut enc = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("zenj-screen") });
        let scene_drawn = match s.engine.as_mut() {
            Some(e) => {
                e.render(&mut enc, &view, FrameClock { time, frame: s.frame });
                true
            }
            None => false,
        };
        s.frame = s.frame.wrapping_add(1);
        {
            let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("zenj-overlay"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: if scene_drawn { LoadOp::Load } else { LoadOp::Clear(Fill::BLACK) },
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Err(e) = s.text.render(&s.atlas, &s.viewport, &mut pass) {
                tracing::debug!("screens: text: {e}");
            }
        }
        gpu.queue.submit(Some(enc.finish()));
        gpu.queue.present(frame);
        s.atlas.trim();
    }
}

impl ApplicationHandler for Screens {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.screens.is_empty() {
            el.set_control_flow(ControlFlow::Poll);
            self.open(el);
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => self.draw(id),
            WindowEvent::Resized(size) => {
                let Some(gpu) = self.gpu.as_ref() else { return };
                if let Some((_, s)) = self.screens.iter_mut().find(|(w, _)| *w == id) {
                    s.config.width = size.width.max(1);
                    s.config.height = size.height.max(1);
                    s.surface.configure(&gpu.device, &s.config);
                    if let Some(e) = s.engine.as_mut() {
                        let _ = e.resize(s.config.width, s.config.height);
                    }
                    s.corner.set_size(Some(s.config.width as f32 - 80.0), Some(s.config.height as f32));
                    s.big.set_size(Some(s.config.width as f32), Some(s.config.height as f32));
                }
            }
            // The screens close only with the process: a stray close on the
            // party's monitors is not obeyed.
            WindowEvent::CloseRequested => {}
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        for (_, s) in &self.screens {
            s.window.request_redraw();
        }
    }
}
