import torch
import torch._dynamo as dynamo
from transformers import AutoModelForSequenceClassification
import time


def benchmark():
    torch._dynamo.config.verbose = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.backends.cuda.matmul.allow_tf32 = True
    #torch.set_float32_matmul_precision('high')
    torch.set_float32_matmul_precision('medium')

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model = dynamo.optimize("inductor")(model)
    test_input = torch.randint(2, (1, 128))

    model.to("cuda")
    test_input.to("cuda")

    #model = torch.compile(model, mode="max-autotune", backend="inductor")
    #model = torch.compile(model, backend="inductor")

    for i in range(5):
        begin = time.time()
        model.forward(test_input)
        end = time.time()
        latency = (end - begin) * 1000
        print(f"Warmup latency: {latency} ms")

    begin = time.time()
    for i in range(100):
        out = model.forward(test_input)
    end = time.time()
    latency = (end - begin) / 100 * 1000
    print(f"Final latency: {latency} ms")


if __name__ == "__main__":
    benchmark()
