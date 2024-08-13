import os
import cv2
import random
import albumentations as A
from tqdm import tqdm  # Progress bar için gerekli

# Augmentasyon işlemleri için transformasyonları tanımlayın
all_transforms = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RandomSizedBBoxSafeCrop(width=640, height=640, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
    A.OpticalDistortion(p=0.3),
    A.GridDistortion(p=0.3),
    A.ElasticTransform(p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.5),
    A.ChannelShuffle(p=0.3),
    A.RandomFog(p=0.3),
    A.RandomRain(p=0.3),
    A.RandomSunFlare(p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    A.MotionBlur(blur_limit=5, p=0.1),
    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3)
]


def get_random_transforms():
    num_transforms = random.randint(7, 18)  # Rastgele olarak 7 ile 18 arasında bir sayı seçin
    return A.Compose(random.sample(all_transforms, num_transforms),
                     bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Klasör yolları
base_dir = "C:/test"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
aug_dir = os.path.join(base_dir, "augDatas")

# Yeni klasörü oluştur
os.makedirs(aug_dir, exist_ok=True)
os.makedirs(os.path.join(aug_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(aug_dir, "val"), exist_ok=True)


def is_bbox_valid(bboxes):
    return all(0 <= min(bbox) <= 1 and 0 <= max(bbox) <= 1 for bbox in bboxes)


def augment_images_and_labels(image_dir, aug_image_dir, label_dir, aug_label_dir):
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
                # İlk rastgele augmentasyon işlemlerini seç
                transform = get_random_transforms()

                # Görüntüyü augment et
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented["image"]
                aug_bboxes = augmented["bboxes"]

                # Augmented görüntü ve etiketleri kaydet
                aug_image_filename = f"aug_{image_file}"
                aug_image_path = os.path.join(aug_image_dir, aug_image_filename)

                cv2.imwrite(aug_image_path, aug_image)

                aug_label_filename = f"aug_{image_file.replace('.jpg', '.txt')}"
                aug_label_path = os.path.join(aug_label_dir, aug_label_filename)
                with open(aug_label_path, 'w') as f:
                    for bbox, class_label in zip(aug_bboxes, class_labels):
                        f.write(f"{class_label} " + " ".join([str(x) for x in bbox]) + "\n")

                # İkinci augmentasyon işlemi (7-10 arası bir değer gelirse)
                if 7 <= random.randint(7, 18) <= 10:
                    transform = get_random_transforms()
                    augmented = transform(image=aug_image, bboxes=aug_bboxes, class_labels=class_labels)
                    aug_image = augmented["image"]
                    aug_bboxes = augmented["bboxes"]

                    # Tekrar augment edilen görüntü ve etiketleri kaydet
                    aug_image_filename = f"aug2_{image_file}"
                    aug_image_path = os.path.join(aug_image_dir, aug_image_filename)
                    cv2.imwrite(aug_image_path, aug_image)

                    aug_label_filename = f"aug2_{image_file.replace('.jpg', '.txt')}"
                    aug_label_path = os.path.join(aug_label_dir, aug_label_filename)
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_label in zip(aug_bboxes, class_labels):
                            f.write(f"{class_label} " + " ".join([str(x) for x in bbox]) + "\n")

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

        pbar.update(1)  # Progress bar'ı güncelle

    pbar.close()  # Progress bar'ı kapat


# Train klasörü için augmentasyon işlemleri
augment_images_and_labels(train_dir, os.path.join(aug_dir, "train"), train_dir, os.path.join(aug_dir, "train"))

# Val klasörü için augmentasyon işlemleri
augment_images_and_labels(val_dir, os.path.join(aug_dir, "val"), val_dir, os.path.join(aug_dir, "val"))

print("Augmentation işlemi tamamlandı!")
