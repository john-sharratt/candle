get rid of the saved secret called demo-webhook-secret<|im_end|>
<|im_start|>assistant
<|projection|><think>
Okay, the user wants to delete a credential called "demo-webhook-secret". Let me check the tools available. There's a function called credential_delete that takes a name parameter. The description says it permanently deletes a stored credential. So I need to call that function with the name "demo-webhook-secret".<|projection|> I should make sure the JSON is correctly formatted with the name key. No other parameters are needed since the required field is just the name. Alright, that should do it.
</think>

<tool_call>
<|projection|>{"name": "credential_delete", "arguments": {"name": "demo-webhook-secret"}}
</tool_call><|projection|>