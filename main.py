from PyQt5.QtGui import QMovie, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QDialog, QFileDialog, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
import arayuz
import sys

#pillow kütüphanesinin bilgisayarımızda bulunması lazım #


from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

from google_trans_new import google_translator 



class LoadingScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(480,480)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)

        self.label_animation = QLabel(self)
        self.movie = QMovie("loading.gif")
        self.label_animation.setMovie(self.movie)

        timer = QTimer(self)
        timer.singleShot(3000,self.stopAnimation)

        self.show()

    def startAnimation(self):
        self.movie.start()

    def stopAnimation(self):
        self.movie.stop()
        self.close()


class Pencere(QMainWindow, arayuz.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.loading_screen = LoadingScreen()
        
        try:
            self.model = ResNet50(weights='imagenet')
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model2 = TFGPT2LMHeadModel.from_pretrained("gpt2",
                pad_token_id=self.tokenizer.eos_token_id)
            self.imagePath = ""

        except:
            self.error2()
            
        self.pushButton_3.setEnabled(False)
        self.plainTextEdit.setEnabled(False)

        self.pushButton.clicked.connect(self.browseImage)
        self.pushButton_2.clicked.connect(self.kontrol)
        self.pushButton_3.clicked.connect(self.cevirici_kontrol)

        self.checkBox.stateChanged.connect(lambda:self.tik())
        
        
    def tik(self):
        if self.checkBox.isChecked() == True:
            self.plainTextEdit.setEnabled(True)
            self.pushButton.setEnabled(False)
            self.comboBox.setEnabled(False)
        else:
            self.plainTextEdit.setEnabled(False)
            self.pushButton.setEnabled(True)
            self.comboBox.setEnabled(True)


    # resim Alma
    def browseImage(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "c\\",
                                                "Image files (*.bmp *.cur *.gif *.icns *.ico *.jpeg *.jpg *.pbm *.pgm *.png *.ppm *.svg *.svgz *.tga *.tif *.tiff *.wbmp *.webp *.xbm *.xpm)")
        self.imagePath = fname[0]

        self.image.setPixmap(QtGui.QPixmap(self.imagePath))
        
        if self.imagePath != "":
            #
            self.progressBar.setValue(10)
            #
            self.resimTanima()
            
        

    def kontrol(self):
        if self.imagePath == "" and self.plainTextEdit.toPlainText() == "":
            self.error()
        if self.checkBox.isChecked() == False:
            self.resim_secilen()
        else:
            self.kelime_secilen()
        

    def resimTanima(self):
        img_path = self.imagePath
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        #
        self.progressBar.setValue(70)
        #

        preds = self.model.predict(x)
        
        sonuc = decode_predictions(preds)[0]
        self.nesneler = []
        le = len(sonuc)

        for i in range(le):
            self.nesneler.append(sonuc[i][1])
        
        #isimleri Düzenleme: 2 kelimeli olan nesnelerin "_" değerlerini değiştiririz
        for i in range(len(self.nesneler)):
            self.nesneler[i] = self.nesneler[i].replace("_", " ")
        self.comboBox.addItems(self.nesneler)

        #
        self.label.setText("Resim Yüklendi")
        self.progressBar.setValue(100)
        #

    def resim_secilen(self):
        #
        self.progressBar.setValue(0)
        #    
        text = str(self.comboBox.currentText())
        self.kac_olacak = int(self.comboBox_2.currentText())
        self.uzunluk = int(self.comboBox_3.currentText())
        
        if text != "":
            self.kelimeKurma(text)
        

    def kelime_secilen(self):
        #
        self.progressBar.setValue(0)
        #    
        self.kac_olacak = int(self.comboBox_2.currentText())
        self.uzunluk = int(self.comboBox_3.currentText())
        
        text = self.plainTextEdit.toPlainText()
        text = text.lower()
        
        if text != "":
            self.kelimeKurma(text)
        


    def kelimeKurma(self, yapacagı_kelime):
        input_ids = self.tokenizer.encode(yapacagı_kelime, return_tensors='tf')

        #
        self.progressBar.setValue(10)
        #

        tf.random.set_seed(0)

        #
        self.label.setText("Cümleler Oluşturuluyor Lütfen Bekleyiniz")
        self.progressBar.setValue(70)
        #
        
        sample_outputs = self.model2.generate(
        input_ids,
        do_sample=True, 
        max_length=self.uzunluk, 
        top_k=50,   
        top_p=0.95, 
        num_return_sequences=self.kac_olacak 
        )

        yazı = ()
        for i, sample_output in enumerate(sample_outputs):
            yazı += i,self.tokenizer.decode(sample_output, skip_special_tokens=True)
            
        self.values = '\n'.join(str(v) for v in yazı)

        #
        self.label.setText("Cümleler Oluşturuldu")
        self.progressBar.setValue(100)
        #
        self.textBrowser.setText(self.values)
        self.cevirici_deger = 0

        if self.values != "":
            self.pushButton_3.setEnabled(True)


    def cevirici_kontrol(self):
        if self.cevirici_deger == 0:
            self.ing_cevirici()
        else:
            self.tr_cevirici()
    
    def ing_cevirici(self):
        self.cevirici_deger = 1
        try:
            translator = google_translator()  
            self.translate_text = translator.translate(self.values,lang_src="en",lang_tgt='tr')
        
            self.textBrowser.setText(self.translate_text)
        except:
            self.belirtec = 1
            self.error2()
        

    def tr_cevirici(self):
        self.cevirici_deger = 0
        self.textBrowser.setText(self.values)
    
    def error(self):
        msg = QMessageBox()
        msg.setText("Lütfen Cümle Giriniz")
        msg.exec_()

    def error2(self):
        msg = QMessageBox()
        msg.setWindowTitle("Hata!")
        if self.belirtec == 1:
            msg.setText("Bir Hata Oluştu Lütfen İnternete Bağlanıp Deneyiniz")
        else:
            msg.setText("Yüklemelerde Bir hata Oluştu")
        msg.exec_()
            


app = QApplication(sys.argv)
pencere = Pencere()
pencere.show()
sys.exit(app.exec_())