# albumentation
Bu uygulama ile modelimizi eğitirken ezberden uzaklaşıp daha iyi sonuçlar elde edebileceğiz. Modelimiz gerçek hayattaki durumlara aşina hale gelecektir. Kodlar içerisinde isteğinize bağlı olarak özellikleri ve oranlarını yeniden yazarak değişiklikler yapabilirsiniz. Albumentation işlemleri ile elimizdeki verilerin augmentation edilmiş hallerini yeni klasörümüzde görebileceksiniz. Bu işlemlerin uygulanması sırasında JPG ve TXT dosyaları birbirleri ile uyumlu şekilde güncellenecektir. <br>
<br>
Benim kodları uyguladığım klasörün dosya biçimi şu şekildedir: <br>
```
C:/test 
├── train 
│   ├── imgTr1.jpg 
│   ├── imgTr1.txt 
│   ├── imgTr2.jpg 
│   ├── imgTr2.txt 
│   ├── ... 
│   ├── ... 
│   └── ... 
└── val 
    ├── imgVal1.jpg 
    ├── imgVal1.txt 
    ├── imgVal2.jpg 
    ├── imgVal2.txt 
    ├── ... 
    ├── ... 
    └── ... 
```
Uygulama bu şekilde olan klasör yapılarına uygun çalışacaktır. 
<br>
Kodu çalıştırdığımız zaman işlem tamamlandığında görünüş şu şekilde olacaktır:
```
C:/test 
├── train 
│   ├── imgTr1.jpg 
│   ├── imgTr1.txt 
│   ├── imgTr2.jpg 
│   ├── imgTr2.txt 
│   ├── ... 
│   ├── ... 
│   └── ... 
├── val 
│   ├── imgVal1.jpg 
│   ├── imgVal1.txt 
│   ├── imgVal2.jpg 
│   ├── imgVal2.txt 
│   ├── ... 
│   ├── ... 
│   └── ... 
└── augDatas
    ├── train 
    │   ├── aug_img1Tr.jpg 
    │   ├── aug_img1Tr.txt 
    │   ├── aug_img2Tr.jpg 
    │   ├── aug_img2Tr.txt 
    │   ├── ... 
    │   ├── ... 
    │   └── ... 
    └── val 
        ├── aug_imgVal1.jpg 
        ├── aug_imgVal1.txt 
        ├── aug_imgVal2.jpg 
        ├── aug_imgVal2.txt 
        ├── ... 
        ├── ... 
        └── ... 
```
