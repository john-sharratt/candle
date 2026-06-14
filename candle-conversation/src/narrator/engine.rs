use crate::{
    handle::TokenDecoder,
    narrator::{narrate_system_prompt, text_to_inputs_streaming, ConverterMode, NarratorInput},
    ConversationEngine, Sequence, SequenceConfig,
};

pub struct NarratorEngine {
    pub conversation: Sequence,
}

impl NarratorEngine {
    pub fn new(engine: &ConversationEngine, mut config: SequenceConfig) -> anyhow::Result<Self> {
        let system_prompt = narrate_system_prompt();
        config.context_window_turns = 4;

        let mut conversation = engine.new_conversation(&system_prompt, config)?;
        conversation.tree_mut().set_max_turns(4);
        Ok(Self { conversation })
    }

    pub fn next(&mut self, inputs: Vec<NarratorInput>) -> anyhow::Result<String> {
        let r1 = self
            .conversation
            .send_turn(&serde_json::to_string(&inputs)?)?;

        Ok(r1.text)
    }

    /// Convert the character's response to compact waypoints, then feed those
    /// waypoints to the narrator via actual inference.
    ///
    /// This is the correct way to update the narrator's context after a character
    /// responds.  The previous approach — `insert_turn(json, char_text)` — stored
    /// raw first-person character prose as the narrator's *assistant* output,
    /// teaching the narrator that it "generated" first-person dialogue.  After
    /// a few turns this contaminated the context and caused output corruption.
    ///
    /// By calling `self.next(waypoints)` the narrator produces its own coherent
    /// second-person prose of what the character said/did, which is stored as the
    /// assistant turn.  The resulting prose is returned to the caller; the caller
    /// may display it or discard it.
    pub fn insert_character_response_streaming(
        &mut self,
        char_text: &str,
        persona: &str,
        engine: &ConversationEngine,
        converter_config: SequenceConfig,
        decoder: &TokenDecoder,
    ) -> anyhow::Result<String> {
        let mode = ConverterMode::Response(persona);
        let waypoints =
            text_to_inputs_streaming(char_text, mode, 3, engine, converter_config, decoder)
                .map_err(|e| anyhow::anyhow!("waypoint conversion failed: {e}"))?;
        // Run the narrator on the character's waypoints so it produces its own
        // coherent prose, which gets stored as the assistant turn — not raw
        // first-person text.
        self.next(waypoints)
    }

    /// Close the narrator conversation and release resources.
    pub fn close(self) -> anyhow::Result<()> {
        self.conversation.close().map_err(Into::into)
    }
}
