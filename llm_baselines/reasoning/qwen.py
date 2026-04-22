import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import pandas as pd
import json
from qwen_vl_utils import process_vision_info
import torch
from reasoning.utils import match_to_class


class QwenVLReasoner():
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to('cuda')
        
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    
    def generate_text(self, img, prompt):
        """ Generate text for a given img and prompt """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image","image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0]



    def accident_spatial_reasoning(self, img, prompt=None):
        if prompt is None:
            prompt = (
                "The scene depicts a traffic accident with one or more cars colliding. "
                "Point to the car accident. "
                "Give the exact coordinates as: <point x='...' y='...'>accident</point>"
            )
        generated_text = self.generate_text(img, prompt)
        point = self.parse_point(generated_text)
        return point, generated_text


    def accident_temporal_reasoning(self, imgs, prompt=None):
        if prompt is None:
            prompt = (
                "Is there a traffic accident or collision? "
                "Yes or No answer only."
            )
    
        preds_raw = {}
        for i, img in imgs:
            preds_raw[i] = self.generate_text(img, prompt)
    
        preds = {k: 'yes' in v.strip().lower() for k, v in preds_raw.items()}
        preds = pd.Series(preds)
        frame_ids = preds[preds].index
        if len(frame_ids) != 0:
            frame_id = frame_ids[0]
        else:
            frame_id = None
        
        return frame_id, preds_raw


    def accident_cause_reasoning(self, img, prompt):
        generated_text = self.generate_text(img, prompt)
        label = match_to_class(generated_text)
        return label, generated_text



    def parse_point(self, text):
        """
        Parses the first <points> tag and returns a (x, y) tuple of ints.
        Returns (None, None) if no match is found.
        """
        pattern = r'<points[^>]*x1="([\d.]+)"[^>]*y1="([\d.]+)"(?:[^>]*alt="([^"]*)")?[^>]*>([^<]*)</points>'
        match = re.search(pattern, text)
        if not match:
            return (None, None)

        x, y, alt, inner_text = match.groups()
        x, y = int(float(x)), int(float(y))
        return (x, y)