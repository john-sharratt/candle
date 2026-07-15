drop the credential named legacy-smtp-password, we don't use it anymore<|im_end|>
<|im_start|>assistant
<|projection|><think>
Okay, the user wants to delete a credential called "legacy-smtp-password". Let me check the tools available. There's a function called credential_delete that takes a name parameter. The description says it permanently deletes a stored credential. The user mentioned they don't use it anymore, so this fits. I need<|projection|> to make sure the name is exact. The user provided the name directly, so I can call the tool with {"name": "legacy-smtp-password"}. No other steps needed here. Just trigger the tool call.
</think>

<tool_call>
<|projection|>{"name": "credential_delete", "arguments": {"name": "legacy-smtp-password"}}
</tool_call><|projection|>