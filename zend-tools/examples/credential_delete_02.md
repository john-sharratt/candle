revoke the SSH key credential named staging-ssh-key<|im_end|>
<|im_start|>assistant
<|projection|><think>
Okay, the user wants to revoke the SSH key credential named "staging-ssh-key". Let me check the available tools. There's a function called credential_delete which requires the exact name of the credential. The user provided the name directly, so I should call credential_delete with the name parameter set to "<|projection|>staging-ssh-key". No need to list credentials first since the name is given. Just make sure the JSON structure is correct.
</think>

<tool_call>
<|projection|>{"name": "credential_delete", "arguments": {"name": "staging-ssh-key"}}
</tool_call><|projection|>