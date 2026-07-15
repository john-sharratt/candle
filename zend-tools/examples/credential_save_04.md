remember the credential named stripe-secret with value sk_live_51HxYzAbCdEfG<|im_end|>
<|im_start|>assistant
<|projection|><think>
Okay, the user wants to save a credential called "stripe-secret" with the value "sk_live_51HxYzAbCdEfG". Let me check the available tools.

The tool provided is credential_save. The parameters needed are name, secret, type, and others like username, passphrase<|projection|>, etc., which might be optional.

The user didn't specify the type, but the secret starts with "sk_live_", which is typically an API key for Stripe. However, the tool's documentation says that for http_bearer, it's an API token. So maybe the type should be http_bearer. But<|projection|> the tool can infer the type from the secret. Let me check the tool's description again.

The tool's description says: "Infer it from the secret: an API token (e.g. one starting `sk-`, `ghp_`, `sk_live_`, `xoxb-`) is `<|projection|>http_bearer`". So in this case, the type would be http_bearer.

So the parameters would be:
- name: stripe-secret
<|projection|>- secret: sk_live_51HxYzAbCdEfG
- type: http_bearer

Other parameters like username, domain, etc., are<|projection|> not provided, so they can be omitted unless required. The required fields are name, secret, and type.

So the JSON call should include those three parameters. Let me structure that.
</think>

<tool_call>
<|projection|>{"name": "credential_save", "arguments": {"name": "stripe-secret", "secret": "sk_live_51HxYzAbCdEfG", "type": "http_bearer"}}
</tool_call><|projection|>