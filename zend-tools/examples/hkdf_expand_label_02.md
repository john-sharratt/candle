derive a 16-byte TLS 1.3 key with SHA-256 from the hex PRK aabbccddeeff00112233445566778899 with label 'key'<|im_end|>
<|im_start|>assistant
<|projection|><think>
Okay, let me try to figure out how to do this. The user wants to derive a 16-byte TLS 1.3 key using SHA-256 from a given PRK. The label is 'key'. 

First, I remember that TLS 1.3 uses HKDF-<|projection|>Expand-Label for deriving keys. The tool provided is hkdf_expand_label. So I need to use that.

The parameters required are algorithm, context_hex (optional), label, length, and prk_hex. The user provided the PRK as a hex string: aabbccddeeff00112<|projection|>233445566778899. The label is 'key', and the length is 16 bytes. The algorithm is SHA-256.

Wait, the label in the tool's description says that the 'tls13 ' prefix is added automatically. So I<|projection|> shouldn't include that. Just the label 'key' is enough.

So the arguments should be:
- algorithm: sha256
<|projection|>- label: key
- length: 16
- prk_hex: aabbccddeeff00112233445566<|projection|>778899

I don't think context_hex is needed here unless there's a transcript hash, which the user didn't mention. So that can be omitted or set to null.

Let me double-check the parameters. The required fields are algorithm, label, length, and prk_hex. All are<|projection|> provided. The context_hex is optional. So the tool call should include those four parameters.
</think>

<tool_call>
<|projection|>{"name": "hkdf_expand_label", "arguments": {"algorithm": "sha256", "label": "key", "length": 16, "prk_hex": "aabbccddeeff00112233445566778899"}}
</tool_call><|projection|>