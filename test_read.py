from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset(
        "nielsr/funsd-layoutlmv3",
    )


def show_image(num=0, dpi=300, name_image='image.png'):
    image = dataset["train"][num]["image"]
    plt.axis("off")
    plt.imshow(image)
    plt.savefig(name_image, dpi=dpi)


def show_image_w_box(num=0, dpi=300, name_image='image.png'):
    image = dataset["train"][num]["image"]
    plt.axis("off")
    plt.imshow(image)
    bbox = dataset["train"][num]["bboxes"]
    x_max, y_max = image.size
    for box in bbox:
        x1, y1, x2, y2 = box
        x1 = x1*x_max/1000
        x2 = x2*x_max/1000
        y1 = y1*y_max/1000 
        y2 = y2*y_max/1000 
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color="blue", linewidth=0.2)
    plt.plot([0, 0, x_max, x_max, 0], [0, y_max, y_max, 0, 0], color='red', linewidth=0.2)
    plt.savefig(name_image, dpi=dpi)


show_image_w_box(num=45)
print("piche")