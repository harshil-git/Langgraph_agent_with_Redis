import boto3
import json


bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")
inference_profile_arn = "us.meta.llama3-1-8b-instruct-v1:0"


def llm_inference(question,context):

    #context = "\n\n".join(context_paragraphs)
    prompt = f"""
                You are a helpful, truthful assistant.

                Answer the question using ONLY the information in the context below.
                If the context does not contain enough information, respond exactly with:
                "I don't know based on the provided context."

                Context:
                {context}

                Question:
                {question}

                Answer ONLY this question one time in one concise paragraph. 
        """


    response = bedrock.invoke_model(
            modelId=inference_profile_arn,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": prompt,
                "temperature": 0.3,
                "top_p": 0.9,
                "max_gen_len": 256,
          }))

    model_output = json.loads(response["body"].read())
    return model_output["generation"].strip()

    
def judge_eval(documents,answer):
    judge_prompt = f"""
                <s>[INST]<<SYS>>
                You are honest judge, judging given answer groundness in given documents. Always judge as helpfully as possible only using the documents and answer text provided. 
                Does the answer stay grounded in the provided documents? give your judged reply with YES/NO only. 
    
                <</SYS>>

                DOCUMENTS:/n/n {documents} /n/n
                Answer: {answer}

                judge reply: [/INST]"""

    response = bedrock.invoke_model(
            modelId=inference_profile_arn,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": judge_prompt,
                "temperature": 0.3,
                "top_p": 0.9,
                "max_gen_len": 512
            }))

    model_output = json.loads(response["body"].read())
    return model_output["generation"].strip()



