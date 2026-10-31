/* The substrate inspector — a lazy tree over one tenant's substrate.
 *
 * The overview is deliberately light (storage + counts + layer metadata); the
 * system prompt, each layer's conversations and each conversation's forest load
 * on expand and cache. A collapsed node holds no DOM, so a layer of thousands
 * of conversations costs nothing until it is opened.
 *
 * The data is the host's (`host::substrate_views`), which every tenant serves
 * over its own substrate: its line in `GET /v1/substrate`, and its views under
 * `/v1/substrate/<tenant>/`. The page that mounts this owns the look: every
 * class this module writes is `prefix` + its name, so a console whose own
 * stylesheet already means something by `.row` or `.chip` styles the
 * inspector's under names of their own.
 *
 * The poll refreshes the cheap header in place. When the counts change it
 * OFFERS a reload rather than performing one — rebuilding would collapse
 * whatever the reader has open and throw away its fetched sub-tree. */

import { memberWith } from './aggregate.js';

const POLL_MS = 6000;

// ── formatting ──────────────────────────────────────────────────────────────
function fmtBytes(b){
  if(b==null) return '—';
  if(b<1024) return b+' B';
  if(b<1048576) return (b/1024).toFixed(1)+' KB';
  if(b<1073741824) return (b/1048576).toFixed(1)+' MB';
  return (b/1073741824).toFixed(2)+' GB';
}
function fmtN(n){ return (n==null?0:n).toLocaleString(); }
function fmtTok(n){
  n=n||0;
  if(n>=1e6) return (n/1e6).toFixed(n>=1e7?0:1)+'M';
  return n>=1000 ? (n/1000).toFixed(n>=10000?0:1)+'k' : ''+n;
}
function shortId(s){ return s && s.length>10 ? s.slice(0,8)+'…' : (s||''); }
function escapeHtml(s){ return (s==null?'':''+s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

async function getJSON(path){
  const r=await fetch(path,{cache:'no-store',headers:{accept:'application/json'}});
  if(!r.ok) throw new Error(r.status);
  return r.json();
}

/* Mount the inspector.
 *
 *   tenant   whose substrate: `zend`, `npcd`.
 *   prefix   prepended to every class the inspector writes; `''` writes the
 *            bare names.
 *   mock     draw the fixture below instead of asking the daemon.
 *   hosts    { kpis, sysprompt, layers }: where each part goes. The segment
 *            bar is placed straight after `kpis`.
 *   live     { set(state, text), onClick(fn | null) }: the page's live
 *            indicator. `state` is `live`, `stale` or `changed`.
 *
 * Returns `{ stop }`, which ends the poll. */
export function mountSubstrate({ tenant, prefix, mock, hosts, live }) {
  const P = prefix || '';
  // A class string with every name prefixed; unchanged when there is no prefix.
  const cls = (s) => (P ? s.split(' ').filter(Boolean).map((n) => P + n).join(' ') : s);

  // ── tiny DOM helper ───────────────────────────────────────────────────────
  function h(tag, attrs, children){
    const e=document.createElement(tag);
    if(attrs) for(const k in attrs){
      if(k==='class') e.className=cls(attrs[k]);
      else if(k==='text') e.textContent=attrs[k];
      else if(k==='html') e.innerHTML=attrs[k];
      else if(k.slice(0,2)==='on') e.addEventListener(k.slice(2), attrs[k]);
      else if(attrs[k]!=null) e.setAttribute(k, attrs[k]);
    }
    if(children!=null){ (Array.isArray(children)?children:[children]).forEach(c=>{
      if(c==null) return; e.appendChild(typeof c==='string'?document.createTextNode(c):c);
    });}
    return e;
  }
  const chev = () => h('span',{class:'chev',html:'<svg viewBox="0 0 10 10" width="10" height="10" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 1 L7 5 L3 9"/></svg>'});
  function badge(c, txt){ return h('span',{class:'badge '+c,text:txt}); }

  // A collapsible card. `bodyFn(body)` fills the body each time it OPENS; the
  // body is emptied when it CLOSES — so a collapsed node holds no DOM, and
  // bodyFn does its (possibly async) fetch only on expand. Fetched data is
  // cached in `S` so re-opening rebuilds from cache without another round-trip.
  function group(headNodes, bodyFn, opts){
    opts = opts||{};
    const card = h('div',{class:'grp'});
    if(opts.accentColor) card.style.borderLeftColor = opts.accentColor;
    const hdr = h('div',{class:'hdr'}, [chev(), ...headNodes]);
    const body = h('div',{class:'body'});
    hdr.addEventListener('click',()=>{
      if(card.classList.toggle(P+'open')) bodyFn(body);
      else body.replaceChildren();   // reduce DOM when closed
    });
    card.appendChild(hdr); card.appendChild(body);
    return card;
  }
  function spinner(txt){ return h('div',{class:'empty',html:'<span class="'+cls('spin')+'"></span> '+escapeHtml(txt||'loading…')}); }
  // True while the card owning `node` is still open — an async bodyFn must
  // check this after its await before writing, or a fetch that resolves after
  // the user collapsed the card would render into (and leave stale DOM in) a
  // closed node. Handles teardown too: a detached node has no card ancestor, so
  // it reads shut.
  function stillOpen(node){ const c = node.closest && node.closest('.'+P+'grp'); return !!c && c.classList.contains(P+'open'); }

  // ── state ─────────────────────────────────────────────────────────────────
  const S = { data:null, lastCounts:null, sp:null, toolsData:null, layerCache:{}, tlCache:{} };

  // ── lazy loaders (fetch-on-expand, cached) ────────────────────────────────
  // The tenant's own views live under `/v1/substrate/<tenant>/`; the overview
  // and the tool catalog are the host's, with the tenant's part named by it.
  const VIEWS = '/v1/substrate/'+tenant;
  async function loadSystemPrompt(){ if(!S.sp) S.sp = mock?mockSystemPrompt():await getJSON(VIEWS+'/system-prompt'); return S.sp; }
  async function loadTools(){ if(!S.toolsData) S.toolsData = mock?mockTools():await getJSON('/v1/tools').then((b)=>({ tools:(b.tools||[]).filter((t)=>t.tenant===tenant) })); return S.toolsData; }
  async function loadLayer(name){ if(!S.layerCache[name]) S.layerCache[name] = mock?mockLayer(name):await getJSON(VIEWS+'/layer/'+encodeURIComponent(name)); return S.layerCache[name]; }
  async function loadTimeline(tl){ if(!S.tlCache[tl]) S.tlCache[tl] = mock?mockTimeline(tl):await getJSON(VIEWS+'/timeline/'+encodeURIComponent(tl)); return S.tlCache[tl]; }

  // ── rendering: KPIs ───────────────────────────────────────────────────────
  function kpi(k, v, unit, x, c){
    return h('div',{class:'kpi'},[
      h('div',{class:'k',text:k}),
      h('div',{class:'v'+(c?' '+c:'')}, unit?[document.createTextNode(v), h('small',{text:unit})]:[document.createTextNode(v)]),
      x?h('div',{class:'x',text:x}):null,
    ]);
  }
  function renderKPIs(d){
    const st=d.storage, c=d.counts;
    const active = st.segments.find(s=>s.active);
    const dead = st.dead_ratio;
    const deadCls = dead==null?'':(dead>=0.5?'crit':dead>=0.25?'warn':'ok');
    hosts.kpis.replaceChildren(
      kpi('substrate size', fmtBytes(st.total_bytes), null, st.segments.length+' segment'+(st.segments.length===1?'':'s')),
      kpi('active segment', active?('#'+active.id):'—', null, active?fmtBytes(active.bytes):''),
      kpi('dead / reclaimable', dead==null?'—':(dead*100).toFixed(1), dead==null?null:'%', dead==null?'compacting…':(dead>=0.5?'compaction due':'healthy'), deadCls),
      kpi('sealed tokens', fmtTok(c.token_count), null, 'turns + shared sections, once each'),
      kpi('KV chunk records', fmtN(st.live_chunks), null, '32 tokens × 1 layer each'),
      kpi('conversations', fmtN(c.conversation_count), null, fmtN(c.timeline_count)+' timelines'),
      kpi('prompt sections', fmtN(c.section_count)),
    );
  }

  // segmented-log bar. Sealed segments alternate two shades, which a page sets
  // as `--seg-a` / `--seg-b`.
  const SEG_A = 'var(--seg-a, #25455a)', SEG_B = 'var(--seg-b, #1c3546)';
  function renderSegBar(st){
    const total = st.total_bytes||1;
    const bar = h('div',{class:'segbar'});
    st.segments.forEach((s,i)=>{
      const span=h('span'); span.style.width=Math.max(1,(s.bytes/total*100))+'%';
      span.style.background = s.active?'var(--accent)':(i%2?SEG_A:SEG_B);
      span.title = 'seg #'+s.id+(s.active?' (active)':' (sealed)')+' — '+fmtBytes(s.bytes);
      bar.appendChild(span);
    });
    const legend=h('div',{class:'seglegend'}, st.segments.map(s=>{
      const i=h('i'); i.style.background=s.active?'var(--accent)':SEG_A;
      return h('span',{},[i, document.createTextNode('#'+s.id+' '+(s.active?'active · ':'sealed · ')+fmtBytes(s.bytes))]);
    }));
    return h('div',{},[bar, legend]);
  }

  // ── rendering: system prompt (fetched on expand) ──────────────────────────
  function renderSysPrompt(d){
    const target = d.layers.find(l=>l.name===d.target_layer);
    const dials = (target && target.dials) ? target.dials : [];
    const head=[
      h('span',{class:'title'},[document.createTextNode('shared system prompt')]),
      h('div',{class:'meta'},[ badge('peak', 'target: '+(d.target_layer||'—')) ]
        .concat(dials.map(x=>badge('', x)))),
    ];
    const card = group(head, async (body)=>{
      body.appendChild(spinner());
      let sp; try{ sp = await loadSystemPrompt(); }
      catch(e){ if(stillOpen(body)) body.replaceChildren(h('div',{class:'empty',text:'Failed to load: '+e})); return; }
      if(!stillOpen(body)) return;   // collapsed while fetching
      body.replaceChildren();
      if(sp.items && sp.items.length){
        body.appendChild(h('div',{class:'desc',text:'Ordered system-prompt structure — the framing the assembler wraps every turn in:'}));
        body.appendChild(h('div',{class:'chips'}, sp.items.map(x=>h('span',{class:'chip',text:x}))));
      }
      if(sp.sections && sp.sections.length){
        body.appendChild(h('div',{class:'desc',text:'Always-emit framing sections — expand to read the authored text tokenized into each:'}));
        sp.sections.forEach(s=> body.appendChild(sectionCard(s)));
      }
      if(sp.tool_count){ body.appendChild(toolsCard(sp.tool_count)); }
    });
    card.style.borderLeftColor='var(--c-sys)';
    hosts.sysprompt.replaceChildren(card);
  }

  // A framing section that expands to reveal its authored text (torn down on close).
  function sectionCard(s){
    const wrap=h('div',{class:'turn k-normal'});
    const hdr=h('div',{class:'thdr'},[
      chev(),
      h('span',{class:'tlabel mono',text:s.name}),
      h('div',{class:'tmeta'},[ badge('', fmtTok(s.tokens)+' tok'), badge('', s.blocks+' blk') ]),
    ]);
    const bd=h('div',{class:'tbody'});
    hdr.addEventListener('click',()=>{
      if(wrap.classList.toggle(P+'open')){
        if(s.content && s.content.trim()) bd.appendChild(h('pre',{class:'body',text:s.content}));
        else bd.appendChild(h('div',{class:'empty',text:'(no authored content)'}));
      } else bd.replaceChildren();
    });
    wrap.appendChild(hdr); wrap.appendChild(bd);
    return wrap;
  }

  // The tools collection — the catalog itself is fetched only when this expands.
  function toolsCard(count){
    const head=[
      h('span',{class:'title'},[document.createTextNode('tools'), h('span',{class:'mono',text:' · collection'})]),
      h('div',{class:'meta'},[ badge('', count+' members') ]),
    ];
    const tcard=group(head, async (tb)=>{
      tb.appendChild(spinner('loading catalog…'));
      let td; try{ td = await loadTools(); }
      catch(e){ if(stillOpen(tb)) tb.replaceChildren(h('div',{class:'empty',text:'Failed to load: '+e})); return; }
      if(!stillOpen(tb)) return;
      tb.replaceChildren();
      const risky = td.tools.filter(t=>t.high_risk).length;
      tb.appendChild(h('div',{class:'desc',text:'The live tool catalog projected into this collection'+(risky?(' — '+risky+' high-risk'):'')+'. BDP selects a subset per turn.'}));
      td.tools.forEach(t=> tb.appendChild(toolRow(t)));
    });
    tcard.style.borderLeftColor='var(--c-think)';
    tcard.style.marginTop='10px';
    return tcard;
  }
  function toolRow(t){
    return h('div',{class:'row'},[
      h('span',{class:'dot'}),
      h('div',{style:'min-width:0;flex:1'},[
        h('div',{class:'nm',text:t.name}),
        t.description?h('div',{style:'color:var(--ink-dim);font-size:12px;margin-top:2px;overflow-wrap:anywhere',text:t.description}):null,
      ]),
      t.high_risk?h('div',{class:'rmeta'},[badge('arch','high-risk')]):null,
    ]);
  }

  // ── rendering: layers (conversations fetched on expand) ───────────────────
  function renderLayers(d){
    hosts.layers.replaceChildren(...d.layers.map(l=>layerCard(l, d)));
  }
  function layerCard(l, d){
    const isTarget = l.name===d.target_layer;
    // Diagnostic kill switch: exclude this layer from projection assembly
    // (in-memory only, clears on restart). `card` is referenced at click time,
    // by which point the const below is assigned.
    const tbtn = h('button',{
      title:'Toggle this layer in the projection (diagnostic, in-memory — clears on restart)',
      style:'cursor:pointer;font:inherit;font-size:11px;padding:1px 8px;border-radius:10px;border:1px solid #555;color:#fff;background:'+(l.disabled?'#b04a4a':'#2f6f4f'),
      onclick: async (e)=>{
        e.stopPropagation();
        try{
          const r = await fetch(VIEWS+'/layer/'+encodeURIComponent(l.name)+'/toggle',{method:'POST'});
          const j = await r.json();
          l.disabled = j.disabled;
          tbtn.style.background = j.disabled ? '#b04a4a' : '#2f6f4f';
          tbtn.textContent = j.disabled ? 'disabled' : 'enabled';
          card.style.opacity = j.disabled ? '0.45' : '';
        }catch(err){ console.error('layer toggle failed', err); }
      }
    },[document.createTextNode(l.disabled?'disabled':'enabled')]);
    const head=[
      h('span',{class:'title mono'},[document.createTextNode(l.name)]),
      h('div',{class:'meta'},[
        l.conversation_count?badge('on', l.conversation_count+' conv'):badge('off','empty'),
        l.tokens?badge('', fmtTok(l.tokens)+' tok'):null,
        isTarget?badge('peak','target'):null,
        badge('', 'win '+fmtTok(l.window)),
        badge('', 'pri '+l.priority),
        tbtn,
      ].filter(Boolean)),
    ];
    const card = group(head, async (body)=>{
      if(l.description) body.appendChild(h('div',{class:'desc',text:l.description}));
      if(l.dials && l.dials.length){
        body.appendChild(h('div',{class:'desc',text:'System-prompt dials (overrides of the shared prompt when this layer is the target):'}));
        body.appendChild(h('div',{class:'chips'}, l.dials.map(x=>h('span',{class:'chip',text:x}))));
      }
      if(l.groups && l.groups.length){
        body.appendChild(h('div',{class:'chips'}, l.groups.map(g=>
          h('span',{class:'chip',html:'<b>'+escapeHtml(g.name)+'</b> · '+escapeHtml(g.selection)}))));
      }
      if(!l.conversation_count){ body.appendChild(h('div',{class:'empty',text:'No conversations target this layer yet.'})); return; }
      const listHost=h('div'); body.appendChild(listHost);
      listHost.appendChild(spinner('loading conversations…'));
      let ld; try{ ld = await loadLayer(l.name); }
      catch(e){ if(stillOpen(listHost)) listHost.replaceChildren(h('div',{class:'empty',text:'Failed to load: '+e})); return; }
      if(!stillOpen(listHost)) return;
      listHost.replaceChildren();
      if(!ld.conversations.length){ listHost.appendChild(h('div',{class:'empty',text:'No conversations.'})); return; }
      ld.conversations.forEach(c=> listHost.appendChild(convCard(c)));
    });
    if(isTarget) card.style.borderLeftColor='var(--accent)';
    if(l.disabled) card.style.opacity='0.45';
    return card;
  }

  // What tells an ingested unit's row from its namesakes. A file reading names
  // the commit it read the file at — the version, which many branches share. A
  // folder names the branches that list it its way: one `candle/` row per
  // distinct listing reads as ten identical rows without them. A long branch
  // list shows its first few and a count, with the whole list on hover.
  const BRANCHES_SHOWN = 3;
  const COMMIT_SHOWN = 10;
  function versionLine(c){
    if(c.commit) return h('span',{class:'branches',title:c.commit,text:'@ '+c.commit.slice(0, COMMIT_SHOWN)});
    const branches = c.branches;
    if(!branches || !branches.length) return null;
    const shown = branches.slice(0, BRANCHES_SHOWN).join(', ');
    const more = branches.length > BRANCHES_SHOWN ? ' +'+(branches.length-BRANCHES_SHOWN)+' more' : '';
    return h('span',{class:'branches',title:branches.join(', '),text:'@ '+shown+more});
  }

  function convCard(c){
    const head=[
      h('span',{class:'title'},[
        // An ingested unit's repository, left of the path inside it — the way
        // the tools address it (`repo: candle, path: CLAUDE.md`).
        c.repo ? h('span',{class:'repo',text:c.repo}) : null,
        document.createTextNode(c.label && c.label.trim() ? c.label : '(untitled)'),
        versionLine(c),
      ].filter(Boolean)),
      h('div',{class:'meta'},[
        c.archived?badge('arch','archived'):null,
        c.summary_nodes?badge('sum', c.summary_nodes+' summary'):null,
        badge('', c.turn_count+' turns'),
        badge('', fmtTok(c.tokens)+' tok'),
        c.conv_id?badge('', shortId(c.conv_id)):null,
      ].filter(Boolean)),
    ];
    return group(head, async (body)=>{
      body.appendChild(spinner('loading forest…'));
      let tl; try{ tl = await loadTimeline(c.timeline); }
      catch(e){ if(stillOpen(body)) body.replaceChildren(h('div',{class:'empty',text:'Failed to load: '+e})); return; }
      if(!stillOpen(body)) return;
      body.replaceChildren();
      if(!tl){ body.appendChild(h('div',{class:'empty',text:'Timeline not found.'})); return; }
      renderForest(body, tl);
    }, {accentColor:'var(--line2)'});
  }

  function renderForest(host, tl){
    const meta=h('div',{class:'chips'},[
      h('span',{class:'chip',html:'<b>'+escapeHtml(tl.layer||'—')+'</b> / '+escapeHtml(tl.group||'—')}),
      h('span',{class:'chip',html:'<b>'+fmtTok(tl.total_tokens)+'</b> tokens'}),
      h('span',{class:'chip',html:'<b>'+tl.turn_count+'</b> turns'}),
      tl.peaks.length?h('span',{class:'chip',html:'<b>'+tl.peaks.length+'</b> forest peaks'}):null,
      tl.couplings.length?h('span',{class:'chip',html:'<b>'+tl.couplings.length+'</b> exchanges'}):null,
    ].filter(Boolean));
    host.appendChild(meta);
    tl.turns.forEach(t=>host.appendChild(turnCard(t)));
  }

  function turnCard(t){
    const kindLabel = t.kind==='sot'?'summary of turns':t.kind==='sos'?'summary of summaries':'turn';
    const card=h('div',{class:'turn k-'+t.kind});
    const thdr=h('div',{class:'thdr'},[
      chev(),
      h('span',{class:'idx',text:'#'+t.index}),
      h('span',{class:'tlabel',text:kindLabel}),
      h('div',{class:'tmeta'},[
        t.peak?badge('peak','peak'):null,
        t.no_think?badge('think','no-think'):null,
        t.coupled?badge('','⇄ coupled'):null,
        t.height?badge(t.kind==='sos'?'sos':'sot','h'+t.height):null,
        badge('', fmtTok(t.tokens)+' tok'),
      ].filter(Boolean)),
    ]);
    const tbody=h('div',{class:'tbody'});
    thdr.addEventListener('click',()=>{
      if(card.classList.toggle(P+'open')) buildTurnBody(tbody, t);
      else tbody.replaceChildren();   // reduce DOM when closed
    });
    card.appendChild(thdr); card.appendChild(tbody);
    return card;
  }

  function buildTurnBody(host, t){
    if(t.children && t.children.length){
      host.appendChild(h('div',{class:'chips'},[
        h('span',{class:'chip',html:'compresses turns <b>'+t.children.join(', ')+'</b>'})
      ]));
    }
    // Readable content, primary. The user/assistant halves and the K/V segment
    // layout describe the SAME tokens, so the segments go in a collapsed
    // sub-panel below rather than duplicating the text inline.
    const userLbl = t.kind==='normal' ? 'user' : 'scope';
    const asstLbl = t.kind==='normal' ? 'assistant' : 'summary';
    let anyText=false;
    if(t.user && t.user.trim()){
      host.appendChild(h('div',{class:'seg'},[h('div',{class:'lbl',text:userLbl}), h('pre',{class:'body user',text:t.user})]));
      anyText=true;
    }
    if(t.assistant && t.assistant.trim()){
      host.appendChild(h('div',{class:'seg'},[h('div',{class:'lbl',text:asstLbl}), h('pre',{class:'body asst',text:t.assistant})]));
      anyText=true;
    }
    const segs = (t.layout && t.layout.segments) || [];
    if(!anyText && !segs.length){
      host.appendChild(h('div',{class:'empty',text:'(no decoded body)'}));
    }
    if(segs.length){
      const head=[
        h('span',{class:'title',text:'K/V layout'}),
        h('div',{class:'meta'},[badge('', segs.length+' segments')]),
      ];
      const kv=group(head,(b)=>{ b.appendChild(renderSegments(segs)); });
      kv.style.borderLeftColor='var(--c-glue)'; kv.style.marginTop='9px';
      host.appendChild(kv);
    }
  }

  // Colorized K/V segment layout as a <pre> (real vs ethereal glue, user,
  // thinking, assistant) — shown inside the collapsed "K/V layout" panel.
  function renderSegments(segs){
    const kindColor={user:'var(--c-user)',assistant:'var(--c-asst)',thinking:'var(--c-think)',glue:'var(--c-glue)'};
    const pre=h('pre',{class:'body'});
    segs.forEach(s=>{
      const kind=(s.kind||'glue').toLowerCase();
      const line=h('span'); line.style.color=kindColor[kind]||'var(--c-glue)';
      const ethereal = s.kv==null;
      if(ethereal){ line.style.opacity='0.5'; line.style.fontStyle='italic'; }
      const txt = s.text!=null ? s.text
        : (typeof s.marker==='string' ? '⟐ '+s.marker : '['+kind+']');
      line.textContent = txt + '\n';
      pre.appendChild(line);
    });
    return pre;
  }

  // ── data + polling ────────────────────────────────────────────────────────
  let segbarHost = null;
  function renderAll(d){
    S.data=d;
    renderKPIs(d);
    // Segment bar right under the KPIs (created once).
    if(!segbarHost){ segbarHost=h('div'); hosts.kpis.after(segbarHost); }
    segbarHost.replaceChildren(renderSegBar(d.storage));
    renderSysPrompt(d);
    renderLayers(d);
    S.lastCounts=JSON.stringify(d.counts);
  }

  async function refresh(){
    try{
      // The overview is intentionally light (storage + counts + layer
      // metadata); conversations, sections and the tool catalog load
      // per-expansion. So this poll stays cheap and never loads the whole
      // substrate up front. It is the tenant's line in the host's overview of
      // every attached substrate, `null` until the tenant can read its
      // substrate. Through a clustered gateway, a member's that holds it.
      const d = mock ? mockOverview() : await getJSON('/v1/substrate').then((body)=>{
        const z = ((memberWith(body,tenant)||{}).tenants||{})[tenant];
        if(!z) throw new Error('503');
        return z;
      });
      if(!S.data){ renderAll(d); live.set('live','live'); return; }
      // Live: refresh KPIs + segment bar in place. If structure changed, offer
      // a reload rather than rebuild — rebuilding would collapse whatever the
      // reader has open (and drop its fetched sub-tree).
      renderKPIs(d);
      segbarHost.replaceChildren(renderSegBar(d.storage));
      S.data.storage=d.storage;
      if(JSON.stringify(d.counts)!==S.lastCounts){
        live.set('changed','new data · tap to reload');
        live.onClick(()=>{
          S.sp=null; S.toolsData=null; S.layerCache={}; S.tlCache={};
          renderAll(d); live.set('live','live'); live.onClick(null);
        });
      } else {
        // Deltas settled — drop any reload handler left from a prior 'changed'.
        live.onClick(null);
        live.set('live','live');
      }
    }catch(e){
      live.set('stale', (''+e).includes('503')?'model loading…':'reconnecting…');
    }
  }

  refresh();
  const timer = setInterval(refresh, POLL_MS);
  return { stop: () => clearInterval(timer) };
}

// ── mock preview data — mirrors the lazy endpoints ──────────────────────────
function mockOverview(){
  return {
    storage:{ segments:[{id:13,active:false,bytes:4630511616},{id:14,active:true,bytes:1966080000}],
      total_bytes:6596591616, live_chunks:537, dead_ratio:0.18 },
    counts:{ timeline_count:9, conversation_count:5, section_count:142, token_count:137250 },
    target_layer:'dialogue',
    layers:[
      {name:'repo_map',description:'Structural shape of the repository: directory tree, file types, line counts.',
        window:6000,priority:5,groups:[{name:'structure',selection:'top 3'}],dials:['thinking_effort = off','response_length = terse','tools = none'],conversation_count:1,tokens:14880},
      {name:'code_reading',description:'Raw scope-aware code chunks. One turn per (file × scope) chunk.',
        window:6000,priority:25,groups:[{name:'scopes',selection:'top 4'}],dials:['thinking_effort = off','response_length = terse','tools = none'],conversation_count:2,tokens:10430},
      {name:'dialogue',description:'The active developer conversation served to the VS Code extension or web chat.',
        window:16000,priority:100,groups:[{name:'primary_conversation',selection:'recent 16 + top 8'}],
        dials:[],conversation_count:3,tokens:35630},
    ],
  };
}
function mockSystemPrompt(){
  return {
    target_layer:'dialogue',
    items:['system open · template','section tree','history_stance','grounding','tools_overview','tools · collection','system close · template'],
    tool_count:7,
    sections:[
      {name:'history_stance',tokens:41,blocks:2,content:'Earlier turns in the history are your own prior work, even where they appear condensed or summarised. Build on what you already examined, concluded, and decided rather than recapping or re-deriving it.'},
      {name:'grounding',tokens:58,blocks:2,content:'Only speak from what is actually present in the conversation. If something you need has not appeared yet, fetch it with a tool when one fits — otherwise say so plainly. Never fill the gap with an invented value or result.'},
      {name:'tools_overview',tokens:171,blocks:6,content:'You have access to the tools listed below. Every listed tool is available and working right now.\n\nNever guess what a tool can tell you — call it.'},
      {name:'grounding_no_tools',tokens:44,blocks:2,content:'Only speak from what is actually present in the conversation. If something you need has not appeared yet, say so plainly rather than guessing.'},
    ],
  };
}
function mockTools(){
  return { tools:[
    {name:'read_file',description:'Read a file (or a line range) from the workspace.',high_risk:false},
    {name:'code_read',description:'Read a scoped code region — a function, struct, or impl block.',high_risk:false},
    {name:'grep',description:'Search file contents across the workspace with a regex.',high_risk:false},
    {name:'list_dir',description:'List the entries of a directory.',high_risk:false},
    {name:'datetime',description:'Return the current date and time in a given timezone.',high_risk:false},
    {name:'write_file',description:'Create or overwrite a file in the workspace.',high_risk:true},
    {name:'run_shell',description:'Execute a shell command in the workspace.',high_risk:true},
  ]};
}
function mockLayer(name){
  const m={
    repo_map:[
      {timeline:'3300220011009988776',conv_id:'',repo:'candle',label:'/',archived:false,group:'structure',turn_count:2,tokens:14880,summary_nodes:0,branches:['main']},
      {timeline:'3300220011009988777',conv_id:'',repo:'candle',label:'/',archived:false,group:'structure',turn_count:2,tokens:14880,summary_nodes:0,branches:['cline-passthrough','hunt-snapshot','npc-engine','qwen38-moe','stencil-json-repair']},
    ],
    code_reading:[
      {timeline:'2200110099887766554',conv_id:'',repo:'candle',label:'candle-conversation/src/substrate.rs',archived:false,group:'scopes',turn_count:9,tokens:6120,summary_nodes:0,branches:['main','qwen38-moe'],commit:'2122c95d9b8e4f1a3c5d7e9f0a1b2c3d4e5f6a7b'},
      {timeline:'2200110099887766555',conv_id:'',repo:'candle',label:'zend/src/session.rs',archived:false,group:'scopes',turn_count:6,tokens:4310,summary_nodes:0,branches:['main'],commit:'1dda19ce4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c'},
    ],
    dialogue:[
      {timeline:'8421990033112233445',conv_id:'a1b2c3d4e5f6',label:'Trace the substrate redo log replay',archived:false,group:'primary_conversation',turn_count:18,tokens:24310,summary_nodes:3},
      {timeline:'7311882299441100223',conv_id:'99aa88bb77cc',label:'Segmented log compaction design',archived:false,group:'primary_conversation',turn_count:7,tokens:9120,summary_nodes:0},
      {timeline:'5120033441122008765',conv_id:'deadbeef0001',label:'Old spike (archived)',archived:true,group:'primary_conversation',turn_count:4,tokens:2200,summary_nodes:0},
    ],
  };
  return { conversations: m[name] || [] };
}
function mockTimeline(tl){
  return {
    timeline:tl, conv_id:'a1b2c3d4e5f6', label:'Trace the substrate redo log replay', archived:false,
    layer:'dialogue', group:'primary_conversation', total_tokens:24310, peaks:[8], couplings:[5],
    turn_count:4,
    turns:[
      {index:0,kind:'normal',height:0,tokens:120,no_think:false,children:[],coupled:false,peak:false,
        user:'Trace how the substrate redo log gets replayed on daemon boot.',
        assistant:'On boot the daemon opens `substrate`, enumerates seg-*.log files, and runs a single walker pass driving every record into the in-RAM Substrate…',layout:null},
      {index:5,kind:'normal',height:0,tokens:340,no_think:false,children:[],coupled:true,peak:false,
        user:'what is the time?',
        assistant:'<tool_call>{"name":"clock","arguments":{}}</tool_call>',layout:null},
      {index:6,kind:'normal',height:0,tokens:88,no_think:false,children:[],coupled:false,peak:false,
        user:'<tool_response>01:05:23</tool_response>',
        assistant:'It is 1:05:23 AM.',layout:null},
      {index:8,kind:'sot',height:1,tokens:96,no_think:false,children:[0],coupled:false,peak:true,
        user:'redo log replay on boot',
        assistant:'Explained boot-time replay: enumerate segments, single walker pass populates the in-RAM Substrate; torn tail frames are skipped.',layout:null},
    ],
  };
}
