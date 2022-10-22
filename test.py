import binary_tokenizer
import transformers
def main():
    pipeline = binary_tokenizer.pipeline('text-generation', 'gpt2', 'pretrained_adapter')
    print(pipeline(b'hello'))
    pipeline.model.save_pretrained('pretrained_adapter')
main()
