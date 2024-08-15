import os
import cv2
import random
import albumentations
from tqdm import tqdm

# Augmentasyon işlemleri için transformasyonları tanımlayın
all_transforms = [
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomBrightnessContrast(p=0.2),
    albumentations.Rotate(limit=30, p=0.5),
    albumentations.RandomScale(scale_limit=0.2, p=0.5),
    albumentations.OpticalDistortion(p=0.2),
    albumentations.GridDistortion(p=1),
    albumentations.ElasticTransform(p=0.2),
    albumentations.CLAHE(clip_limit=2.0, p=0.5),
    albumentations.ChannelShuffle(p=0.3),
    albumentations.RandomFog(p=0.1),
    albumentations.RandomRain(p=0.1),
    albumentations.RandomSunFlare(p=0.3),
    albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    albumentations.GaussianBlur(blur_limit=(1, 3), p=0.2),
    albumentations.MotionBlur(blur_limit=3, p=0.2),
    albumentations.ImageCompression(quality_lower=30, quality_upper=60, p=0.3)
]

# Klasör yolları
base_dir = "C:/learningDataYOLOv5"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
aug_dir = os.path.join(base_dir, "augDatas")
crops_dir = os.path.join(base_dir, "crops")

# Yeni klasörleri oluştur
os.makedirs(aug_dir, exist_ok=True)
os.makedirs(os.path.join(aug_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(aug_dir, "val"), exist_ok=True)
os.makedirs(crops_dir, exist_ok=True)
os.makedirs(os.path.join(crops_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(crops_dir, "val"), exist_ok=True)


def is_bbox_valid(bboxes):
    return all(0 <= min(bbox) <= 1 and 0 <= max(bbox) <= 1 for bbox in bboxes)


def crop_image(image, bboxes, class_labels, crop_size=(640, 640), num_crops=10):
    crops = []
    height, width = image.shape[:2]

    for i in range(num_crops):
        x = random.randint(0, width - crop_size[0])
        y = random.randint(0, height - crop_size[1])

        cropped_image = image[y:y + crop_size[1], x:x + crop_size[0]]
        cropped_bboxes = []
        cropped_labels = []

        for bbox, label in zip(bboxes, class_labels):
            bbox_x_min = int(bbox[0] * width)
            bbox_y_min = int(bbox[1] * height)
            bbox_x_max = int(bbox[2] * width)
            bbox_y_max = int(bbox[3] * height)

            if bbox_x_max > x and bbox_x_min < x + crop_size[0] and bbox_y_max > y and bbox_y_min < y + crop_size[1]:
                cropped_bboxes.append([
                    max(0, bbox_x_min - x) / crop_size[0],
                    max(0, bbox_y_min - y) / crop_size[1],
                    min(crop_size[0], bbox_x_max - x) / crop_size[0],
                    min(crop_size[1], bbox_y_max - y) / crop_size[1]
                ])
                cropped_labels.append(label)

        if cropped_bboxes:  # Eğer crop bboxes boş değilse, crop'u ekle
            crops.append((cropped_image, cropped_bboxes, cropped_labels))

    return crops


def augment_images_and_labels(image_dir, aug_image_dir, label_dir, aug_label_dir, crop_image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    pbar = tqdm(total=len(image_files), desc=f"Processing images in {os.path.basename(image_dir)}")

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))

        # Görüntüyü yükle
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Etiketleri yükle
        with open(label_path, 'r') as f:
            labels = f.readlines()

        bboxes = []
        class_labels = []
        for label in labels:
            parts = label.strip().split()
            class_labels.append(int(parts[0]))
            bbox = [float(x) for x in parts[1:]]
            bboxes.append(bbox)

        if is_bbox_valid(bboxes):
            try:
                # Augmentasyon işlemlerini sırayla uygula
                for transform in all_transforms:
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    image = augmented["image"]
                    bboxes = augmented["bboxes"]

                # Augmented görüntü ve etiketleri kaydet
                aug_image_filename = f"aug_{image_file}"
                aug_image_path = os.path.join(aug_image_dir, aug_image_filename)

                cv2.imwrite(aug_image_path, image)

                aug_label_filename = f"aug_{image_file.replace('.jpg', '.txt')}"
                aug_label_path = os.path.join(aug_label_dir, aug_label_filename)
                with open(aug_label_path, 'w') as f:
                    for bbox, class_label in zip(bboxes, class_labels):
                        f.write(f"{class_label} " + " ".join([str(x) for x in bbox]) + "\n")

                # Crop işlemi
                crops = crop_image(image, bboxes, class_labels)
                for idx, (crop_img, crop_bboxes, crop_labels) in enumerate(crops):
                    crop_filename = f"{image_file.replace('.jpg', '')}_Part{idx + 1}.jpg"
                    crop_image_path = os.path.join(crop_image_dir, crop_filename)
                    crop_label_filename = f"{image_file.replace('.jpg', '')}_Part{idx + 1}.txt"
                    crop_label_path = os.path.join(crop_image_dir, crop_label_filename)

                    # Eğer crop işlemi sonucunda oluşan TXT dosyası boş değilse kaydet
                    if crop_bboxes:
                        cv2.imwrite(crop_image_path, crop_img)
                        with open(crop_label_path, 'w') as f:
                            for bbox, label in zip(crop_bboxes, crop_labels):
                                f.write(f"{label} " + " ".join([str(x) for x in bbox]) + "\n")
                    else:
                        # Eğer TXT dosyası boşsa, aynı numarayı tekrar dene
                        idx -= 1

            except Exception as e:
                print(f"Augmentation failed for {image_file}: {e}")
                # Geçersiz bbox olan dosyaları doğrudan kopyala
                aug_image_filename = f"aug_{image_file}"
                aug_image_path = os.path.join(aug_image_dir, aug_image_filename)
                cv2.imwrite(aug_image_path, image)

                aug_label_filename = f"aug_{image_file.replace('.jpg', '.txt')}"
                aug_label_path = os.path.join(aug_label_dir, aug_label_filename)
                with open(label_path, 'r') as f:
                    with open(aug_label_path, 'w') as fw:
                        fw.write(f.read())
        else:
            # Geçersiz bbox olan dosyaları doğrudan kopyala
            aug_image_filename = f"aug_{image_file}"
            aug_image_path = os.path.join(aug_image_dir, aug_image_filename)
            cv2.imwrite(aug_image_path, image)

            aug_label_filename = f"aug_{image_file.replace('.jpg', '.txt')}"
            aug_label_path = os.path.join(aug_label_dir, aug_label_filename)
            with open(label_path, 'r') as f:
                with open(aug_label_path, 'w') as fw:
                    fw.write(f.read())

        pbar.update(1)

    pbar.close()


# Train klasörü için augmentasyon ve crop işlemleri
augment_images_and_labels(train_dir, os.path.join(aug_dir, "train"), train_dir, os.path.join(aug_dir, "train"),
                          os.path.join(crops_dir, "train"))

# Val klasörü için augmentasyon ve crop işlemleri
augment_images_and_labels(val_dir, os.path.join(aug_dir, "val"), val_dir, os.path.join(aug_dir, "val"),
                          os.path.join(crops_dir, "val"))

print("Augmentation ve crop işlemi tamamlandı!")
