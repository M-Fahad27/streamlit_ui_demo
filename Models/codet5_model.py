from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class CodeExplainer:
    def __init__(
        self,
        model_name: str = "Salesforce/codet5p-220m",
        device: str | None = None,
        max_input_length: int = 512,
        max_output_token: int = 256,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_token = max_output_token

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(
        self, code: str, language: str = "python", explain_kind: str | None = None
    ):
        header = (
            f"### Instruction:\nExplain the following {language} code in clear English."
        )
        if explain_kind:
            header += f" Provide a {explain_kind} explanation."
        prompt = f"{header}\n\n### Code:\n{code}\n\n### Explanation:\n"
        return prompt

    def explain(
        self,
        code: str,
        language: str = "python",
        explain_kind: str | None = None,
        num_beams: int = 4,
        temperature: float = 0.7,
        do_sample: bool = False,
    ) -> str:
        prompt = self._build_prompt(
            code=code, language=language, explain_kind=explain_kind
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_token,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        explaination = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if explaination.startswith(prompt):
            explaination = explaination[len(prompt) :].strip()
        return explaination.strip()
