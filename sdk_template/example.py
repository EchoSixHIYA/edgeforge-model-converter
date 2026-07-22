from edgeforge_sdk import YoloV10

with YoloV10("model") as model:
    results = model.predict("test.jpg")
    for item in results:
        print(item.as_dict())
    model.draw("test.jpg", results, "result.jpg")
