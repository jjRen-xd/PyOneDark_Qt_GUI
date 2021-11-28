# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_pagesGQRQnd.ui'
##
## Created by: Qt User Interface Compiler version 6.2.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from qt_core import *

class Ui_MainPages(object):
    def setupUi(self, MainPages):
        if not MainPages.objectName():
            MainPages.setObjectName(u"MainPages")
        MainPages.resize(1282, 1068)
        MainPages.setMinimumSize(QSize(80, 50))
        self.main_pages_layout = QVBoxLayout(MainPages)
        self.main_pages_layout.setSpacing(0)
        self.main_pages_layout.setObjectName(u"main_pages_layout")
        self.main_pages_layout.setContentsMargins(5, 5, 5, 5)
        self.pages = QStackedWidget(MainPages)
        self.pages.setObjectName(u"pages")
        self.page_home = QWidget()
        self.page_home.setObjectName(u"page_home")
        self.page_home.setStyleSheet(u"font-size: 14pt")
        self.page_1_layout = QVBoxLayout(self.page_home)
        self.page_1_layout.setSpacing(5)
        self.page_1_layout.setObjectName(u"page_1_layout")
        self.page_1_layout.setContentsMargins(5, 5, 5, 5)
        self.welcome_base = QFrame(self.page_home)
        self.welcome_base.setObjectName(u"welcome_base")
        self.welcome_base.setMinimumSize(QSize(300, 150))
        self.welcome_base.setMaximumSize(QSize(800, 180))
        self.welcome_base.setBaseSize(QSize(0, 0))
        font = QFont()
        font.setPointSize(14)
        self.welcome_base.setFont(font)
        self.welcome_base.setFrameShape(QFrame.NoFrame)
        self.welcome_base.setFrameShadow(QFrame.Raised)
        self.center_page_layout = QVBoxLayout(self.welcome_base)
        self.center_page_layout.setSpacing(10)
        self.center_page_layout.setObjectName(u"center_page_layout")
        self.center_page_layout.setContentsMargins(0, 0, 0, 0)
        self.logo = QFrame(self.welcome_base)
        self.logo.setObjectName(u"logo")
        self.logo.setMinimumSize(QSize(300, 120))
        self.logo.setMaximumSize(QSize(300, 120))
        self.logo.setFrameShape(QFrame.NoFrame)
        self.logo.setFrameShadow(QFrame.Raised)
        self.logo_layout = QVBoxLayout(self.logo)
        self.logo_layout.setSpacing(0)
        self.logo_layout.setObjectName(u"logo_layout")
        self.logo_layout.setContentsMargins(0, 0, 0, 0)

        self.center_page_layout.addWidget(self.logo, 0, Qt.AlignHCenter)

        self.label = QLabel(self.welcome_base)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(True)
        font1.setItalic(False)
        font1.setStrikeOut(False)
        self.label.setFont(font1)
        self.label.setAlignment(Qt.AlignCenter)

        self.center_page_layout.addWidget(self.label)


        self.page_1_layout.addWidget(self.welcome_base, 0, Qt.AlignHCenter|Qt.AlignVCenter)

        self.pages.addWidget(self.page_home)
        self.page_widgets = QWidget()
        self.page_widgets.setObjectName(u"page_widgets")
        self.page_2_layout = QVBoxLayout(self.page_widgets)
        self.page_2_layout.setSpacing(5)
        self.page_2_layout.setObjectName(u"page_2_layout")
        self.page_2_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_area = QScrollArea(self.page_widgets)
        self.scroll_area.setObjectName(u"scroll_area")
        self.scroll_area.setEnabled(True)
        self.scroll_area.setStyleSheet(u"background: transparent;")
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        self.contents = QWidget()
        self.contents.setObjectName(u"contents")
        self.contents.setGeometry(QRect(0, 0, 1262, 1048))
        self.contents.setStyleSheet(u"background: transparent;")
        self.verticalLayout = QVBoxLayout(self.contents)
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.title_label = QLabel(self.contents)
        self.title_label.setObjectName(u"title_label")
        self.title_label.setMaximumSize(QSize(16777215, 40))
        font2 = QFont()
        font2.setPointSize(16)
        self.title_label.setFont(font2)
        self.title_label.setStyleSheet(u"font-size: 16pt")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.title_label)

        self.description_label = QLabel(self.contents)
        self.description_label.setObjectName(u"description_label")
        self.description_label.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
        self.description_label.setWordWrap(True)

        self.verticalLayout.addWidget(self.description_label)

        self.row_1_layout = QHBoxLayout()
        self.row_1_layout.setObjectName(u"row_1_layout")

        self.verticalLayout.addLayout(self.row_1_layout)

        self.row_2_layout = QHBoxLayout()
        self.row_2_layout.setObjectName(u"row_2_layout")

        self.verticalLayout.addLayout(self.row_2_layout)

        self.row_3_layout = QHBoxLayout()
        self.row_3_layout.setObjectName(u"row_3_layout")

        self.verticalLayout.addLayout(self.row_3_layout)

        self.row_4_layout = QVBoxLayout()
        self.row_4_layout.setObjectName(u"row_4_layout")

        self.verticalLayout.addLayout(self.row_4_layout)

        self.row_5_layout = QVBoxLayout()
        self.row_5_layout.setObjectName(u"row_5_layout")

        self.verticalLayout.addLayout(self.row_5_layout)

        self.scroll_area.setWidget(self.contents)

        self.page_2_layout.addWidget(self.scroll_area)

        self.pages.addWidget(self.page_widgets)
        self.page_OP = QWidget()
        self.page_OP.setObjectName(u"page_OP")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.page_OP.sizePolicy().hasHeightForWidth())
        self.page_OP.setSizePolicy(sizePolicy1)
        self.page_OP.setStyleSheet(u"QFrame {\n"
"	font-size: 16pt;\n"
"}")
        self.page_3_layout = QVBoxLayout(self.page_OP)
        self.page_3_layout.setObjectName(u"page_3_layout")
        self.title_label_op = QLabel(self.page_OP)
        self.title_label_op.setObjectName(u"title_label_op")
        self.title_label_op.setMaximumSize(QSize(16777215, 40))
        font3 = QFont()
        font3.setPointSize(26)
        font3.setBold(False)
        font3.setItalic(False)
        self.title_label_op.setFont(font3)
        self.title_label_op.setStyleSheet(u"font: 26pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.title_label_op.setAlignment(Qt.AlignCenter)

        self.page_3_layout.addWidget(self.title_label_op)

        self.label_27 = QLabel(self.page_OP)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setMaximumSize(QSize(16777215, 10))

        self.page_3_layout.addWidget(self.label_27)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_op_data = QLabel(self.page_OP)
        self.label_op_data.setObjectName(u"label_op_data")
        self.label_op_data.setMaximumSize(QSize(16777215, 20))
        self.label_op_data.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_data.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.label_op_data)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.comboBox_op_data = QComboBox(self.page_OP)
        self.comboBox_op_data.addItem("")
        self.comboBox_op_data.setObjectName(u"comboBox_op_data")
        self.comboBox_op_data.setMinimumSize(QSize(150, 40))
        self.comboBox_op_data.setMaximumSize(QSize(100, 16777215))
        font4 = QFont()
        font4.setPointSize(14)
        font4.setBold(False)
        font4.setItalic(False)
        self.comboBox_op_data.setFont(font4)
        self.comboBox_op_data.setAutoFillBackground(False)
        self.comboBox_op_data.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_op_data.setIconSize(QSize(16, 16))
        self.comboBox_op_data.setFrame(True)

        self.horizontalLayout_8.addWidget(self.comboBox_op_data)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_2 = QLabel(self.page_OP)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_7.addWidget(self.label_2)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_op_snr = QLabel(self.page_OP)
        self.label_op_snr.setObjectName(u"label_op_snr")
        self.label_op_snr.setMinimumSize(QSize(20, 0))
        self.label_op_snr.setMaximumSize(QSize(16777215, 20))
        self.label_op_snr.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_10.addWidget(self.label_op_snr)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.comboBox_op_snr = QComboBox(self.page_OP)
        self.comboBox_op_snr.addItem("")
        self.comboBox_op_snr.setObjectName(u"comboBox_op_snr")
        self.comboBox_op_snr.setMinimumSize(QSize(0, 40))
        self.comboBox_op_snr.setMaximumSize(QSize(100, 16777215))
        self.comboBox_op_snr.setFont(font4)
        self.comboBox_op_snr.setAutoFillBackground(False)
        self.comboBox_op_snr.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_op_snr.setIconSize(QSize(16, 16))
        self.comboBox_op_snr.setFrame(True)

        self.horizontalLayout_13.addWidget(self.comboBox_op_snr)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_3 = QLabel(self.page_OP)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_11.addWidget(self.label_3)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_op_sample = QLabel(self.page_OP)
        self.label_op_sample.setObjectName(u"label_op_sample")
        self.label_op_sample.setMinimumSize(QSize(20, 0))
        self.label_op_sample.setMaximumSize(QSize(16777215, 20))
        self.label_op_sample.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_sample.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_12.addWidget(self.label_op_sample)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.comboBox_op_sample = QComboBox(self.page_OP)
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.addItem("")
        self.comboBox_op_sample.setObjectName(u"comboBox_op_sample")
        self.comboBox_op_sample.setMinimumSize(QSize(120, 40))
        self.comboBox_op_sample.setMaximumSize(QSize(100, 16777215))
        self.comboBox_op_sample.setFont(font4)
        self.comboBox_op_sample.setAutoFillBackground(False)
        self.comboBox_op_sample.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_op_sample.setIconSize(QSize(16, 16))
        self.comboBox_op_sample.setFrame(True)

        self.horizontalLayout_6.addWidget(self.comboBox_op_sample)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_4 = QLabel(self.page_OP)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_5.addWidget(self.label_4)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_5)

        self.layout_op_data_open = QHBoxLayout()
        self.layout_op_data_open.setObjectName(u"layout_op_data_open")

        self.horizontalLayout_2.addLayout(self.layout_op_data_open)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.label_5 = QLabel(self.page_OP)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_15.addWidget(self.label_5)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_15)


        self.page_3_layout.addLayout(self.horizontalLayout_2)

        self.label_59 = QLabel(self.page_OP)
        self.label_59.setObjectName(u"label_59")
        self.label_59.setMaximumSize(QSize(16777215, 10))
        self.label_59.setStyleSheet(u"")
        self.label_59.setFrameShape(QFrame.HLine)
        self.label_59.setFrameShadow(QFrame.Plain)

        self.page_3_layout.addWidget(self.label_59)

        self.scroll_area_op = QScrollArea(self.page_OP)
        self.scroll_area_op.setObjectName(u"scroll_area_op")
        self.scroll_area_op.setStyleSheet(u"background: transparent;")
        self.scroll_area_op.setFrameShape(QFrame.NoFrame)
        self.scroll_area_op.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area_op.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area_op.setWidgetResizable(True)
        self.contents_op = QWidget()
        self.contents_op.setObjectName(u"contents_op")
        self.contents_op.setGeometry(QRect(0, 0, 1254, 910))
        self.contents_op.setLayoutDirection(Qt.LeftToRight)
        self.contents_op.setStyleSheet(u"background: transparent;")
        self.verticalLayout_2 = QVBoxLayout(self.contents_op)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_29 = QHBoxLayout()
        self.horizontalLayout_29.setSpacing(2)
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalLayout_33 = QHBoxLayout()
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.pic_op_data_1 = QLabel(self.contents_op)
        self.pic_op_data_1.setObjectName(u"pic_op_data_1")
        self.pic_op_data_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_data_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_33.addWidget(self.pic_op_data_1)


        self.horizontalLayout_29.addLayout(self.horizontalLayout_33)

        self.horizontalLayout_32 = QHBoxLayout()
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.pic_op_data_2 = QLabel(self.contents_op)
        self.pic_op_data_2.setObjectName(u"pic_op_data_2")
        self.pic_op_data_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_data_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_32.addWidget(self.pic_op_data_2)


        self.horizontalLayout_29.addLayout(self.horizontalLayout_32)

        self.horizontalLayout_73 = QHBoxLayout()
        self.horizontalLayout_73.setObjectName(u"horizontalLayout_73")
        self.pic_op_data_3 = QLabel(self.contents_op)
        self.pic_op_data_3.setObjectName(u"pic_op_data_3")
        self.pic_op_data_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_data_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_73.addWidget(self.pic_op_data_3)


        self.horizontalLayout_29.addLayout(self.horizontalLayout_73)

        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.pic_op_data_4 = QLabel(self.contents_op)
        self.pic_op_data_4.setObjectName(u"pic_op_data_4")
        self.pic_op_data_4.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_data_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_31.addWidget(self.pic_op_data_4)


        self.horizontalLayout_29.addLayout(self.horizontalLayout_31)


        self.verticalLayout_2.addLayout(self.horizontalLayout_29)

        self.horizontalLayout_34 = QHBoxLayout()
        self.horizontalLayout_34.setSpacing(2)
        self.horizontalLayout_34.setObjectName(u"horizontalLayout_34")
        self.horizontalLayout_36 = QHBoxLayout()
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.pic_op_data_5 = QLabel(self.contents_op)
        self.pic_op_data_5.setObjectName(u"pic_op_data_5")
        self.pic_op_data_5.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_data_5.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_36.addWidget(self.pic_op_data_5)


        self.horizontalLayout_34.addLayout(self.horizontalLayout_36)

        self.horizontalLayout_74 = QHBoxLayout()
        self.horizontalLayout_74.setObjectName(u"horizontalLayout_74")
        self.pic_op_data_6 = QLabel(self.contents_op)
        self.pic_op_data_6.setObjectName(u"pic_op_data_6")
        self.pic_op_data_6.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_data_6.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_74.addWidget(self.pic_op_data_6)


        self.horizontalLayout_34.addLayout(self.horizontalLayout_74)

        self.horizontalLayout_35 = QHBoxLayout()
        self.horizontalLayout_35.setObjectName(u"horizontalLayout_35")
        self.pic_op_data_7 = QLabel(self.contents_op)
        self.pic_op_data_7.setObjectName(u"pic_op_data_7")
        self.pic_op_data_7.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_data_7.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_35.addWidget(self.pic_op_data_7)


        self.horizontalLayout_34.addLayout(self.horizontalLayout_35)

        self.horizontalLayout_37 = QHBoxLayout()
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.pic_op_data_8 = QLabel(self.contents_op)
        self.pic_op_data_8.setObjectName(u"pic_op_data_8")
        self.pic_op_data_8.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_data_8.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_37.addWidget(self.pic_op_data_8)


        self.horizontalLayout_34.addLayout(self.horizontalLayout_37)


        self.verticalLayout_2.addLayout(self.horizontalLayout_34)

        self.label_6 = QLabel(self.contents_op)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMaximumSize(QSize(16777215, 10))
        self.label_6.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_6.setFrameShape(QFrame.HLine)
        self.label_6.setFrameShadow(QFrame.Plain)
        self.label_6.setLineWidth(1)

        self.verticalLayout_2.addWidget(self.label_6)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.label_op_base_model = QLabel(self.contents_op)
        self.label_op_base_model.setObjectName(u"label_op_base_model")
        self.label_op_base_model.setMaximumSize(QSize(170, 16777215))
        self.label_op_base_model.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_base_model.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_18.addWidget(self.label_op_base_model)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_18)

        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.label_7 = QLabel(self.contents_op)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_22.addWidget(self.label_7)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_22)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.label_op_shot_base = QLabel(self.contents_op)
        self.label_op_shot_base.setObjectName(u"label_op_shot_base")
        self.label_op_shot_base.setMaximumSize(QSize(150, 16777215))
        self.label_op_shot_base.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_shot_base.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_21.addWidget(self.label_op_shot_base)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_21)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.comboBox_op_shot_base = QComboBox(self.contents_op)
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.addItem("")
        self.comboBox_op_shot_base.setObjectName(u"comboBox_op_shot_base")
        self.comboBox_op_shot_base.setMinimumSize(QSize(0, 40))
        self.comboBox_op_shot_base.setMaximumSize(QSize(100, 16777215))
        self.comboBox_op_shot_base.setFont(font4)
        self.comboBox_op_shot_base.setAutoFillBackground(False)
        self.comboBox_op_shot_base.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_op_shot_base.setIconSize(QSize(16, 16))
        self.comboBox_op_shot_base.setFrame(True)

        self.horizontalLayout_20.addWidget(self.comboBox_op_shot_base)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_20)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_10 = QLabel(self.contents_op)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_19.addWidget(self.label_10)


        self.horizontalLayout_17.addLayout(self.horizontalLayout_19)

        self.layout_op_test_base = QVBoxLayout()
        self.layout_op_test_base.setObjectName(u"layout_op_test_base")

        self.horizontalLayout_17.addLayout(self.layout_op_test_base)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_17)

        self.label_9 = QLabel(self.contents_op)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setMaximumSize(QSize(5, 16777215))
        self.label_9.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_9.setFrameShape(QFrame.VLine)
        self.label_9.setFrameShadow(QFrame.Plain)

        self.horizontalLayout_9.addWidget(self.label_9)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_op_adv_model = QLabel(self.contents_op)
        self.label_op_adv_model.setObjectName(u"label_op_adv_model")
        self.label_op_adv_model.setMinimumSize(QSize(0, 0))
        self.label_op_adv_model.setMaximumSize(QSize(100, 16777215))
        self.label_op_adv_model.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_adv_model.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_op_adv_model)


        self.horizontalLayout_16.addLayout(self.verticalLayout_4)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.label_op_choice_method = QLabel(self.contents_op)
        self.label_op_choice_method.setObjectName(u"label_op_choice_method")
        self.label_op_choice_method.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_choice_method.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_23.addWidget(self.label_op_choice_method)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_23)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.comboBox_op_method = QComboBox(self.contents_op)
        self.comboBox_op_method.addItem("")
        self.comboBox_op_method.addItem("")
        self.comboBox_op_method.addItem("")
        self.comboBox_op_method.setObjectName(u"comboBox_op_method")
        self.comboBox_op_method.setMinimumSize(QSize(130, 40))
        self.comboBox_op_method.setMaximumSize(QSize(100, 16777215))
        font5 = QFont()
        font5.setPointSize(12)
        font5.setBold(False)
        font5.setItalic(False)
        self.comboBox_op_method.setFont(font5)
        self.comboBox_op_method.setAutoFillBackground(False)
        self.comboBox_op_method.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 12pt \"Segoe UI\";")
        self.comboBox_op_method.setIconSize(QSize(16, 16))
        self.comboBox_op_method.setFrame(True)

        self.horizontalLayout_26.addWidget(self.comboBox_op_method)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_26)

        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.label_op_shot_adv = QLabel(self.contents_op)
        self.label_op_shot_adv.setObjectName(u"label_op_shot_adv")
        self.label_op_shot_adv.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_shot_adv.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_25.addWidget(self.label_op_shot_adv)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_25)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.comboBox_op_shot_adv = QComboBox(self.contents_op)
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.addItem("")
        self.comboBox_op_shot_adv.setObjectName(u"comboBox_op_shot_adv")
        self.comboBox_op_shot_adv.setMinimumSize(QSize(0, 40))
        self.comboBox_op_shot_adv.setMaximumSize(QSize(100, 16777215))
        self.comboBox_op_shot_adv.setFont(font4)
        self.comboBox_op_shot_adv.setAutoFillBackground(False)
        self.comboBox_op_shot_adv.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_op_shot_adv.setIconSize(QSize(16, 16))
        self.comboBox_op_shot_adv.setFrame(True)

        self.horizontalLayout_24.addWidget(self.comboBox_op_shot_adv)


        self.horizontalLayout_16.addLayout(self.horizontalLayout_24)

        self.layout_op_test_adv = QHBoxLayout()
        self.layout_op_test_adv.setObjectName(u"layout_op_test_adv")

        self.horizontalLayout_16.addLayout(self.layout_op_test_adv)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_16)


        self.verticalLayout_2.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_30 = QHBoxLayout()
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.layout_op_acc_base = QHBoxLayout()
        self.layout_op_acc_base.setObjectName(u"layout_op_acc_base")

        self.horizontalLayout_30.addLayout(self.layout_op_acc_base)

        self.label_8 = QLabel(self.contents_op)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setMaximumSize(QSize(5, 16777215))
        self.label_8.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_8.setFrameShape(QFrame.VLine)
        self.label_8.setFrameShadow(QFrame.Plain)

        self.horizontalLayout_30.addWidget(self.label_8)

        self.layout_op_acc_adv = QHBoxLayout()
        self.layout_op_acc_adv.setObjectName(u"layout_op_acc_adv")

        self.horizontalLayout_30.addLayout(self.layout_op_acc_adv)


        self.verticalLayout_2.addLayout(self.horizontalLayout_30)

        self.layout_op_imageShow = QWidget(self.contents_op)
        self.layout_op_imageShow.setObjectName(u"layout_op_imageShow")
        self.horizontalLayout = QHBoxLayout(self.layout_op_imageShow)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.layout_op_imageShow_1 = QHBoxLayout()
        self.layout_op_imageShow_1.setObjectName(u"layout_op_imageShow_1")
        self.pic_op_matrix1 = QLabel(self.layout_op_imageShow)
        self.pic_op_matrix1.setObjectName(u"pic_op_matrix1")
        self.pic_op_matrix1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_matrix1.setAlignment(Qt.AlignCenter)

        self.layout_op_imageShow_1.addWidget(self.pic_op_matrix1)


        self.horizontalLayout.addLayout(self.layout_op_imageShow_1)

        self.layout_op_imageShow_2 = QHBoxLayout()
        self.layout_op_imageShow_2.setObjectName(u"layout_op_imageShow_2")
        self.pic_op_matrix2 = QLabel(self.layout_op_imageShow)
        self.pic_op_matrix2.setObjectName(u"pic_op_matrix2")
        self.pic_op_matrix2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_op_matrix2.setAlignment(Qt.AlignCenter)

        self.layout_op_imageShow_2.addWidget(self.pic_op_matrix2)


        self.horizontalLayout.addLayout(self.layout_op_imageShow_2)


        self.verticalLayout_2.addWidget(self.layout_op_imageShow)

        self.scroll_area_op.setWidget(self.contents_op)

        self.page_3_layout.addWidget(self.scroll_area_op)

        self.pages.addWidget(self.page_OP)
        self.page_TL = QWidget()
        self.page_TL.setObjectName(u"page_TL")
        self.verticalLayout_3 = QVBoxLayout(self.page_TL)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.title_label_op_2 = QLabel(self.page_TL)
        self.title_label_op_2.setObjectName(u"title_label_op_2")
        self.title_label_op_2.setMaximumSize(QSize(16777215, 40))
        self.title_label_op_2.setFont(font3)
        self.title_label_op_2.setStyleSheet(u"font: 26pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.title_label_op_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.title_label_op_2)

        self.label_28 = QLabel(self.page_TL)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setMaximumSize(QSize(16777215, 10))

        self.verticalLayout_3.addWidget(self.label_28)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_op_data_2 = QLabel(self.page_TL)
        self.label_op_data_2.setObjectName(u"label_op_data_2")
        self.label_op_data_2.setMaximumSize(QSize(16777215, 50))
        self.label_op_data_2.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_data_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_14.addWidget(self.label_op_data_2)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.comboBox_tl_method = QComboBox(self.page_TL)
        self.comboBox_tl_method.addItem("")
        self.comboBox_tl_method.addItem("")
        self.comboBox_tl_method.addItem("")
        self.comboBox_tl_method.addItem("")
        self.comboBox_tl_method.addItem("")
        self.comboBox_tl_method.setObjectName(u"comboBox_tl_method")
        self.comboBox_tl_method.setMinimumSize(QSize(160, 40))
        self.comboBox_tl_method.setMaximumSize(QSize(100, 50))
        self.comboBox_tl_method.setFont(font4)
        self.comboBox_tl_method.setAutoFillBackground(False)
        self.comboBox_tl_method.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_tl_method.setIconSize(QSize(16, 16))
        self.comboBox_tl_method.setFrame(True)

        self.horizontalLayout_27.addWidget(self.comboBox_tl_method)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_27)

        self.horizontalLayout_28 = QHBoxLayout()
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.label_11 = QLabel(self.page_TL)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setMinimumSize(QSize(0, 80))
        self.label_11.setMaximumSize(QSize(16777215, 100))

        self.horizontalLayout_28.addWidget(self.label_11)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_28)

        self.horizontalLayout_38 = QHBoxLayout()
        self.horizontalLayout_38.setObjectName(u"horizontalLayout_38")
        self.label_op_snr_2 = QLabel(self.page_TL)
        self.label_op_snr_2.setObjectName(u"label_op_snr_2")
        self.label_op_snr_2.setMaximumSize(QSize(16777215, 50))
        self.label_op_snr_2.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_38.addWidget(self.label_op_snr_2)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_38)

        self.horizontalLayout_39 = QHBoxLayout()
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.comboBox_tl_dataset = QComboBox(self.page_TL)
        self.comboBox_tl_dataset.addItem("")
        self.comboBox_tl_dataset.setObjectName(u"comboBox_tl_dataset")
        self.comboBox_tl_dataset.setMinimumSize(QSize(150, 40))
        self.comboBox_tl_dataset.setMaximumSize(QSize(120, 50))
        self.comboBox_tl_dataset.setFont(font4)
        self.comboBox_tl_dataset.setAutoFillBackground(False)
        self.comboBox_tl_dataset.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_tl_dataset.setIconSize(QSize(16, 16))
        self.comboBox_tl_dataset.setFrame(True)

        self.horizontalLayout_39.addWidget(self.comboBox_tl_dataset)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_39)

        self.horizontalLayout_40 = QHBoxLayout()
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.label_12 = QLabel(self.page_TL)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setMaximumSize(QSize(16777215, 50))

        self.horizontalLayout_40.addWidget(self.label_12)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_40)

        self.horizontalLayout_41 = QHBoxLayout()
        self.horizontalLayout_41.setObjectName(u"horizontalLayout_41")
        self.label_op_snr_3 = QLabel(self.page_TL)
        self.label_op_snr_3.setObjectName(u"label_op_snr_3")
        self.label_op_snr_3.setMaximumSize(QSize(16777215, 50))
        self.label_op_snr_3.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_41.addWidget(self.label_op_snr_3)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_41)

        self.horizontalLayout_42 = QHBoxLayout()
        self.horizontalLayout_42.setObjectName(u"horizontalLayout_42")
        self.comboBox_tl_snr = QComboBox(self.page_TL)
        self.comboBox_tl_snr.addItem("")
        self.comboBox_tl_snr.setObjectName(u"comboBox_tl_snr")
        self.comboBox_tl_snr.setMinimumSize(QSize(80, 40))
        self.comboBox_tl_snr.setMaximumSize(QSize(120, 50))
        self.comboBox_tl_snr.setFont(font4)
        self.comboBox_tl_snr.setAutoFillBackground(False)
        self.comboBox_tl_snr.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_tl_snr.setIconSize(QSize(16, 16))
        self.comboBox_tl_snr.setFrame(True)

        self.horizontalLayout_42.addWidget(self.comboBox_tl_snr)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_42)

        self.horizontalLayout_44 = QHBoxLayout()
        self.horizontalLayout_44.setObjectName(u"horizontalLayout_44")
        self.label_14 = QLabel(self.page_TL)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setMaximumSize(QSize(16777215, 50))

        self.horizontalLayout_44.addWidget(self.label_14)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_44)


        self.verticalLayout_3.addLayout(self.horizontalLayout_4)

        self.label_58 = QLabel(self.page_TL)
        self.label_58.setObjectName(u"label_58")
        self.label_58.setMaximumSize(QSize(16777215, 10))
        self.label_58.setStyleSheet(u"")
        self.label_58.setFrameShape(QFrame.HLine)

        self.verticalLayout_3.addWidget(self.label_58)

        self.scroll_area_tl = QScrollArea(self.page_TL)
        self.scroll_area_tl.setObjectName(u"scroll_area_tl")
        self.scroll_area_tl.setStyleSheet(u"background: transparent;")
        self.scroll_area_tl.setFrameShape(QFrame.NoFrame)
        self.scroll_area_tl.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area_tl.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area_tl.setWidgetResizable(True)
        self.contents_tl = QWidget()
        self.contents_tl.setObjectName(u"contents_tl")
        self.contents_tl.setGeometry(QRect(0, 0, 1254, 872))
        self.verticalLayout_5 = QVBoxLayout(self.contents_tl)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_43 = QHBoxLayout()
        self.horizontalLayout_43.setObjectName(u"horizontalLayout_43")
        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.label_tran4 = QLabel(self.contents_tl)
        self.label_tran4.setObjectName(u"label_tran4")
        self.label_tran4.setMinimumSize(QSize(0, 40))
        self.label_tran4.setMaximumSize(QSize(16777215, 50))
        self.label_tran4.setFrameShape(QFrame.NoFrame)
        self.label_tran4.setFrameShadow(QFrame.Sunken)
        self.label_tran4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.label_tran4)

        self.pic_tl_source = QLabel(self.contents_tl)
        self.pic_tl_source.setObjectName(u"pic_tl_source")
        self.pic_tl_source.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_tl_source.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.pic_tl_source)


        self.horizontalLayout_43.addLayout(self.verticalLayout_8)

        self.label_31 = QLabel(self.contents_tl)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setMaximumSize(QSize(1, 16777215))
        self.label_31.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_31.setFrameShape(QFrame.VLine)

        self.horizontalLayout_43.addWidget(self.label_31)

        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.label_tran5 = QLabel(self.contents_tl)
        self.label_tran5.setObjectName(u"label_tran5")
        self.label_tran5.setMinimumSize(QSize(0, 40))
        self.label_tran5.setMaximumSize(QSize(16777215, 50))
        self.label_tran5.setFrameShape(QFrame.NoFrame)
        self.label_tran5.setFrameShadow(QFrame.Sunken)
        self.label_tran5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_10.addWidget(self.label_tran5)

        self.pic_tl_target = QLabel(self.contents_tl)
        self.pic_tl_target.setObjectName(u"pic_tl_target")
        self.pic_tl_target.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_tl_target.setAlignment(Qt.AlignCenter)

        self.verticalLayout_10.addWidget(self.pic_tl_target)


        self.horizontalLayout_43.addLayout(self.verticalLayout_10)


        self.verticalLayout_5.addLayout(self.horizontalLayout_43)

        self.label_29 = QLabel(self.contents_tl)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setMaximumSize(QSize(16777215, 10))
        self.label_29.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_29.setFrameShape(QFrame.HLine)

        self.verticalLayout_5.addWidget(self.label_29)

        self.horizontalLayout_45 = QHBoxLayout()
        self.horizontalLayout_45.setObjectName(u"horizontalLayout_45")
        self.verticalLayout_12 = QVBoxLayout()
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.horizontalLayout_47 = QHBoxLayout()
        self.horizontalLayout_47.setObjectName(u"horizontalLayout_47")
        self.label_tran6 = QLabel(self.contents_tl)
        self.label_tran6.setObjectName(u"label_tran6")
        self.label_tran6.setMinimumSize(QSize(0, 50))
        self.label_tran6.setMaximumSize(QSize(250, 16777215))
        self.label_tran6.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_tran6.setFrameShape(QFrame.NoFrame)
        self.label_tran6.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_47.addWidget(self.label_tran6)

        self.comboBox_tl_shot = QComboBox(self.contents_tl)
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.addItem("")
        self.comboBox_tl_shot.setObjectName(u"comboBox_tl_shot")
        self.comboBox_tl_shot.setMinimumSize(QSize(120, 40))
        self.comboBox_tl_shot.setMaximumSize(QSize(100, 16777215))
        self.comboBox_tl_shot.setFont(font4)
        self.comboBox_tl_shot.setAutoFillBackground(False)
        self.comboBox_tl_shot.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_tl_shot.setInsertPolicy(QComboBox.InsertBeforeCurrent)
        self.comboBox_tl_shot.setIconSize(QSize(16, 16))
        self.comboBox_tl_shot.setFrame(True)

        self.horizontalLayout_47.addWidget(self.comboBox_tl_shot)


        self.verticalLayout_12.addLayout(self.horizontalLayout_47)

        self.horizontalLayout_57 = QHBoxLayout()
        self.horizontalLayout_57.setObjectName(u"horizontalLayout_57")
        self.label_tran6_2 = QLabel(self.contents_tl)
        self.label_tran6_2.setObjectName(u"label_tran6_2")
        self.label_tran6_2.setMinimumSize(QSize(0, 50))
        self.label_tran6_2.setMaximumSize(QSize(250, 16777215))
        self.label_tran6_2.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_tran6_2.setFrameShape(QFrame.NoFrame)
        self.label_tran6_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_57.addWidget(self.label_tran6_2)

        self.comboBox_tl_samples = QComboBox(self.contents_tl)
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.addItem("")
        self.comboBox_tl_samples.setObjectName(u"comboBox_tl_samples")
        self.comboBox_tl_samples.setMinimumSize(QSize(120, 40))
        self.comboBox_tl_samples.setMaximumSize(QSize(100, 16777215))
        self.comboBox_tl_samples.setFont(font4)
        self.comboBox_tl_samples.setAutoFillBackground(False)
        self.comboBox_tl_samples.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_tl_samples.setInsertPolicy(QComboBox.InsertBeforeCurrent)
        self.comboBox_tl_samples.setIconSize(QSize(16, 16))
        self.comboBox_tl_samples.setFrame(True)

        self.horizontalLayout_57.addWidget(self.comboBox_tl_samples)


        self.verticalLayout_12.addLayout(self.horizontalLayout_57)

        self.horizontalLayout_48 = QHBoxLayout()
        self.horizontalLayout_48.setObjectName(u"horizontalLayout_48")
        self.label_20 = QLabel(self.contents_tl)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setMinimumSize(QSize(0, 50))
        self.label_20.setMaximumSize(QSize(20, 16777215))

        self.horizontalLayout_48.addWidget(self.label_20)

        self.layout_tl_test_base = QHBoxLayout()
        self.layout_tl_test_base.setObjectName(u"layout_tl_test_base")

        self.horizontalLayout_48.addLayout(self.layout_tl_test_base)

        self.label_37 = QLabel(self.contents_tl)
        self.label_37.setObjectName(u"label_37")
        self.label_37.setMaximumSize(QSize(20, 16777215))

        self.horizontalLayout_48.addWidget(self.label_37)

        self.layout_tl_test_adv = QHBoxLayout()
        self.layout_tl_test_adv.setObjectName(u"layout_tl_test_adv")

        self.horizontalLayout_48.addLayout(self.layout_tl_test_adv)

        self.label_57 = QLabel(self.contents_tl)
        self.label_57.setObjectName(u"label_57")
        self.label_57.setMaximumSize(QSize(20, 16777215))

        self.horizontalLayout_48.addWidget(self.label_57)


        self.verticalLayout_12.addLayout(self.horizontalLayout_48)

        self.label_30 = QLabel(self.contents_tl)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setMaximumSize(QSize(16777215, 10))
        self.label_30.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_30.setFrameShape(QFrame.HLine)

        self.verticalLayout_12.addWidget(self.label_30)

        self.label_tran7 = QLabel(self.contents_tl)
        self.label_tran7.setObjectName(u"label_tran7")
        self.label_tran7.setMinimumSize(QSize(0, 50))
        self.label_tran7.setMaximumSize(QSize(16777215, 50))
        self.label_tran7.setAlignment(Qt.AlignBottom|Qt.AlignLeading|Qt.AlignLeft)

        self.verticalLayout_12.addWidget(self.label_tran7)

        self.horizontalLayout_49 = QHBoxLayout()
        self.horizontalLayout_49.setObjectName(u"horizontalLayout_49")
        self.label_tran8 = QLabel(self.contents_tl)
        self.label_tran8.setObjectName(u"label_tran8")
        self.label_tran8.setMinimumSize(QSize(0, 50))
        self.label_tran8.setFrameShape(QFrame.NoFrame)
        self.label_tran8.setFrameShadow(QFrame.Sunken)
        self.label_tran8.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_49.addWidget(self.label_tran8)

        self.label_16 = QLabel(self.contents_tl)
        self.label_16.setObjectName(u"label_16")

        self.horizontalLayout_49.addWidget(self.label_16)

        self.layout_base_acc = QHBoxLayout()
        self.layout_base_acc.setObjectName(u"layout_base_acc")

        self.horizontalLayout_49.addLayout(self.layout_base_acc)

        self.label_13 = QLabel(self.contents_tl)
        self.label_13.setObjectName(u"label_13")

        self.horizontalLayout_49.addWidget(self.label_13)


        self.verticalLayout_12.addLayout(self.horizontalLayout_49)

        self.horizontalLayout_50 = QHBoxLayout()
        self.horizontalLayout_50.setObjectName(u"horizontalLayout_50")
        self.label_tran8_2 = QLabel(self.contents_tl)
        self.label_tran8_2.setObjectName(u"label_tran8_2")
        self.label_tran8_2.setMinimumSize(QSize(0, 50))
        self.label_tran8_2.setFrameShape(QFrame.NoFrame)
        self.label_tran8_2.setFrameShadow(QFrame.Sunken)
        self.label_tran8_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_50.addWidget(self.label_tran8_2)

        self.label_17 = QLabel(self.contents_tl)
        self.label_17.setObjectName(u"label_17")

        self.horizontalLayout_50.addWidget(self.label_17)

        self.layout_adv_acc = QHBoxLayout()
        self.layout_adv_acc.setObjectName(u"layout_adv_acc")

        self.horizontalLayout_50.addLayout(self.layout_adv_acc)

        self.label_15 = QLabel(self.contents_tl)
        self.label_15.setObjectName(u"label_15")

        self.horizontalLayout_50.addWidget(self.label_15)


        self.verticalLayout_12.addLayout(self.horizontalLayout_50)


        self.horizontalLayout_45.addLayout(self.verticalLayout_12)

        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.layout_base_matrix = QLabel(self.contents_tl)
        self.layout_base_matrix.setObjectName(u"layout_base_matrix")
        self.layout_base_matrix.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.layout_base_matrix.setAlignment(Qt.AlignCenter)

        self.verticalLayout_11.addWidget(self.layout_base_matrix)

        self.label_tran10 = QLabel(self.contents_tl)
        self.label_tran10.setObjectName(u"label_tran10")
        self.label_tran10.setMaximumSize(QSize(16777215, 30))
        self.label_tran10.setFrameShape(QFrame.NoFrame)
        self.label_tran10.setFrameShadow(QFrame.Sunken)
        self.label_tran10.setAlignment(Qt.AlignCenter)

        self.verticalLayout_11.addWidget(self.label_tran10)


        self.horizontalLayout_45.addLayout(self.verticalLayout_11)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.layout_adv_matrix = QLabel(self.contents_tl)
        self.layout_adv_matrix.setObjectName(u"layout_adv_matrix")
        self.layout_adv_matrix.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.layout_adv_matrix.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.layout_adv_matrix)

        self.label_tran11 = QLabel(self.contents_tl)
        self.label_tran11.setObjectName(u"label_tran11")
        self.label_tran11.setMaximumSize(QSize(16777215, 30))
        self.label_tran11.setFrameShape(QFrame.NoFrame)
        self.label_tran11.setFrameShadow(QFrame.Sunken)
        self.label_tran11.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_tran11)


        self.horizontalLayout_45.addLayout(self.verticalLayout_7)


        self.verticalLayout_5.addLayout(self.horizontalLayout_45)

        self.scroll_area_tl.setWidget(self.contents_tl)

        self.verticalLayout_3.addWidget(self.scroll_area_tl)

        self.pages.addWidget(self.page_TL)
        self.page_RL = QWidget()
        self.page_RL.setObjectName(u"page_RL")
        self.verticalLayout_6 = QVBoxLayout(self.page_RL)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.label_61 = QLabel(self.page_RL)
        self.label_61.setObjectName(u"label_61")
        self.label_61.setMaximumSize(QSize(16777215, 10))

        self.verticalLayout_6.addWidget(self.label_61)

        self.title_label_op_3 = QLabel(self.page_RL)
        self.title_label_op_3.setObjectName(u"title_label_op_3")
        self.title_label_op_3.setMaximumSize(QSize(16777215, 40))
        self.title_label_op_3.setFont(font3)
        self.title_label_op_3.setStyleSheet(u"font: 26pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.title_label_op_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_6.addWidget(self.title_label_op_3)

        self.label_32 = QLabel(self.page_RL)
        self.label_32.setObjectName(u"label_32")
        self.label_32.setMaximumSize(QSize(16777215, 10))

        self.verticalLayout_6.addWidget(self.label_32)

        self.label_60 = QLabel(self.page_RL)
        self.label_60.setObjectName(u"label_60")
        self.label_60.setMaximumSize(QSize(16777215, 10))
        self.label_60.setStyleSheet(u"")
        self.label_60.setFrameShape(QFrame.HLine)

        self.verticalLayout_6.addWidget(self.label_60)

        self.scroll_area_rl = QScrollArea(self.page_RL)
        self.scroll_area_rl.setObjectName(u"scroll_area_rl")
        self.scroll_area_rl.setStyleSheet(u"background: transparent;")
        self.scroll_area_rl.setFrameShape(QFrame.NoFrame)
        self.scroll_area_rl.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area_rl.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area_rl.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 1272, 964))
        self.scrollAreaWidgetContents.setStyleSheet(u"background: transparent;")
        self.verticalLayout_9 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_46 = QHBoxLayout()
        self.horizontalLayout_46.setObjectName(u"horizontalLayout_46")
        self.verticalLayout_13 = QVBoxLayout()
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.horizontalLayout_51 = QHBoxLayout()
        self.horizontalLayout_51.setObjectName(u"horizontalLayout_51")
        self.label_36 = QLabel(self.scrollAreaWidgetContents)
        self.label_36.setObjectName(u"label_36")
        self.label_36.setMinimumSize(QSize(0, 80))
        self.label_36.setMaximumSize(QSize(16777215, 150))

        self.horizontalLayout_51.addWidget(self.label_36)

        self.label_op_snr_4 = QLabel(self.scrollAreaWidgetContents)
        self.label_op_snr_4.setObjectName(u"label_op_snr_4")
        self.label_op_snr_4.setMaximumSize(QSize(16777215, 50))
        self.label_op_snr_4.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_51.addWidget(self.label_op_snr_4)

        self.comboBox_rl_dataset = QComboBox(self.scrollAreaWidgetContents)
        self.comboBox_rl_dataset.addItem("")
        self.comboBox_rl_dataset.setObjectName(u"comboBox_rl_dataset")
        self.comboBox_rl_dataset.setMinimumSize(QSize(150, 40))
        self.comboBox_rl_dataset.setMaximumSize(QSize(120, 50))
        self.comboBox_rl_dataset.setFont(font4)
        self.comboBox_rl_dataset.setAutoFillBackground(False)
        self.comboBox_rl_dataset.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_rl_dataset.setIconSize(QSize(16, 16))
        self.comboBox_rl_dataset.setFrame(True)

        self.horizontalLayout_51.addWidget(self.comboBox_rl_dataset)

        self.label_23 = QLabel(self.scrollAreaWidgetContents)
        self.label_23.setObjectName(u"label_23")

        self.horizontalLayout_51.addWidget(self.label_23)

        self.label_op_snr_5 = QLabel(self.scrollAreaWidgetContents)
        self.label_op_snr_5.setObjectName(u"label_op_snr_5")
        self.label_op_snr_5.setMaximumSize(QSize(16777215, 50))
        self.label_op_snr_5.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_5.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_51.addWidget(self.label_op_snr_5)

        self.comboBox_rl_snr = QComboBox(self.scrollAreaWidgetContents)
        self.comboBox_rl_snr.addItem("")
        self.comboBox_rl_snr.setObjectName(u"comboBox_rl_snr")
        self.comboBox_rl_snr.setMinimumSize(QSize(80, 40))
        self.comboBox_rl_snr.setMaximumSize(QSize(120, 50))
        self.comboBox_rl_snr.setFont(font4)
        self.comboBox_rl_snr.setAutoFillBackground(False)
        self.comboBox_rl_snr.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_rl_snr.setIconSize(QSize(16, 16))
        self.comboBox_rl_snr.setFrame(True)

        self.horizontalLayout_51.addWidget(self.comboBox_rl_snr)

        self.label_22 = QLabel(self.scrollAreaWidgetContents)
        self.label_22.setObjectName(u"label_22")

        self.horizontalLayout_51.addWidget(self.label_22)


        self.verticalLayout_13.addLayout(self.horizontalLayout_51)

        self.label_38 = QLabel(self.scrollAreaWidgetContents)
        self.label_38.setObjectName(u"label_38")
        self.label_38.setMaximumSize(QSize(16777215, 10))
        self.label_38.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_38.setFrameShape(QFrame.HLine)

        self.verticalLayout_13.addWidget(self.label_38)

        self.horizontalLayout_52 = QHBoxLayout()
        self.horizontalLayout_52.setObjectName(u"horizontalLayout_52")
        self.horizontalLayout_64 = QHBoxLayout()
        self.horizontalLayout_64.setObjectName(u"horizontalLayout_64")
        self.label_19 = QLabel(self.scrollAreaWidgetContents)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setMinimumSize(QSize(0, 80))
        self.label_19.setMaximumSize(QSize(16777215, 150))

        self.horizontalLayout_64.addWidget(self.label_19)

        self.horizontalLayout_63 = QHBoxLayout()
        self.horizontalLayout_63.setObjectName(u"horizontalLayout_63")
        self.radioButton_rl_ori = QRadioButton(self.scrollAreaWidgetContents)
        self.radioButton_rl_ori.setObjectName(u"radioButton_rl_ori")
        self.radioButton_rl_ori.setMaximumSize(QSize(180, 16777215))
        self.radioButton_rl_ori.setStyleSheet(u"font: 14pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.radioButton_rl_ori.setChecked(True)

        self.horizontalLayout_63.addWidget(self.radioButton_rl_ori)


        self.horizontalLayout_64.addLayout(self.horizontalLayout_63)

        self.label_24 = QLabel(self.scrollAreaWidgetContents)
        self.label_24.setObjectName(u"label_24")

        self.horizontalLayout_64.addWidget(self.label_24)

        self.horizontalLayout_62 = QHBoxLayout()
        self.horizontalLayout_62.setObjectName(u"horizontalLayout_62")
        self.radioButton_rl_all = QRadioButton(self.scrollAreaWidgetContents)
        self.radioButton_rl_all.setObjectName(u"radioButton_rl_all")
        self.radioButton_rl_all.setMaximumSize(QSize(180, 16777215))
        self.radioButton_rl_all.setStyleSheet(u"font: 14pt \"\u5fae\u8f6f\u96c5\u9ed1\";")

        self.horizontalLayout_62.addWidget(self.radioButton_rl_all)


        self.horizontalLayout_64.addLayout(self.horizontalLayout_62)

        self.label_25 = QLabel(self.scrollAreaWidgetContents)
        self.label_25.setObjectName(u"label_25")

        self.horizontalLayout_64.addWidget(self.label_25)

        self.horizontalLayout_61 = QHBoxLayout()
        self.horizontalLayout_61.setObjectName(u"horizontalLayout_61")
        self.radioButton_rl_rei = QRadioButton(self.scrollAreaWidgetContents)
        self.radioButton_rl_rei.setObjectName(u"radioButton_rl_rei")
        self.radioButton_rl_rei.setMaximumSize(QSize(180, 16777215))
        self.radioButton_rl_rei.setStyleSheet(u"font: 14pt \"\u5fae\u8f6f\u96c5\u9ed1\";")

        self.horizontalLayout_61.addWidget(self.radioButton_rl_rei)


        self.horizontalLayout_64.addLayout(self.horizontalLayout_61)

        self.label_21 = QLabel(self.scrollAreaWidgetContents)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setMinimumSize(QSize(0, 100))

        self.horizontalLayout_64.addWidget(self.label_21)


        self.horizontalLayout_52.addLayout(self.horizontalLayout_64)


        self.verticalLayout_13.addLayout(self.horizontalLayout_52)

        self.label_35 = QLabel(self.scrollAreaWidgetContents)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setMaximumSize(QSize(16777215, 10))
        self.label_35.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_35.setFrameShape(QFrame.HLine)

        self.verticalLayout_13.addWidget(self.label_35)

        self.horizontalLayout_56 = QHBoxLayout()
        self.horizontalLayout_56.setObjectName(u"horizontalLayout_56")
        self.label_26 = QLabel(self.scrollAreaWidgetContents)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setMinimumSize(QSize(0, 80))
        self.label_26.setMaximumSize(QSize(16777215, 150))

        self.horizontalLayout_56.addWidget(self.label_26)

        self.horizontalLayout_55 = QHBoxLayout()
        self.horizontalLayout_55.setObjectName(u"horizontalLayout_55")
        self.label_op_snr_12 = QLabel(self.scrollAreaWidgetContents)
        self.label_op_snr_12.setObjectName(u"label_op_snr_12")
        self.label_op_snr_12.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_12.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_55.addWidget(self.label_op_snr_12)

        self.comboBox_rl_samples = QComboBox(self.scrollAreaWidgetContents)
        self.comboBox_rl_samples.addItem("")
        self.comboBox_rl_samples.addItem("")
        self.comboBox_rl_samples.addItem("")
        self.comboBox_rl_samples.addItem("")
        self.comboBox_rl_samples.addItem("")
        self.comboBox_rl_samples.addItem("")
        self.comboBox_rl_samples.addItem("")
        self.comboBox_rl_samples.addItem("")
        self.comboBox_rl_samples.setObjectName(u"comboBox_rl_samples")
        self.comboBox_rl_samples.setMinimumSize(QSize(140, 40))
        self.comboBox_rl_samples.setMaximumSize(QSize(100, 16777215))
        self.comboBox_rl_samples.setFont(font4)
        self.comboBox_rl_samples.setAutoFillBackground(False)
        self.comboBox_rl_samples.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_rl_samples.setInsertPolicy(QComboBox.InsertBeforeCurrent)
        self.comboBox_rl_samples.setIconSize(QSize(16, 16))
        self.comboBox_rl_samples.setFrame(True)

        self.horizontalLayout_55.addWidget(self.comboBox_rl_samples)


        self.horizontalLayout_56.addLayout(self.horizontalLayout_55)

        self.label_34 = QLabel(self.scrollAreaWidgetContents)
        self.label_34.setObjectName(u"label_34")

        self.horizontalLayout_56.addWidget(self.label_34)

        self.layout_rl_test = QHBoxLayout()
        self.layout_rl_test.setObjectName(u"layout_rl_test")

        self.horizontalLayout_56.addLayout(self.layout_rl_test)

        self.label_18 = QLabel(self.scrollAreaWidgetContents)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setMinimumSize(QSize(0, 100))

        self.horizontalLayout_56.addWidget(self.label_18)


        self.verticalLayout_13.addLayout(self.horizontalLayout_56)

        self.layout_rl_acc = QVBoxLayout()
        self.layout_rl_acc.setObjectName(u"layout_rl_acc")

        self.verticalLayout_13.addLayout(self.layout_rl_acc)


        self.horizontalLayout_46.addLayout(self.verticalLayout_13)

        self.verticalLayout_14 = QVBoxLayout()
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_16 = QVBoxLayout()
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.pic_rl_matrix = QLabel(self.scrollAreaWidgetContents)
        self.pic_rl_matrix.setObjectName(u"pic_rl_matrix")
        self.pic_rl_matrix.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_rl_matrix.setAlignment(Qt.AlignCenter)

        self.verticalLayout_16.addWidget(self.pic_rl_matrix)


        self.verticalLayout_14.addLayout(self.verticalLayout_16)

        self.label_tran10_2 = QLabel(self.scrollAreaWidgetContents)
        self.label_tran10_2.setObjectName(u"label_tran10_2")
        self.label_tran10_2.setMaximumSize(QSize(16777215, 30))
        self.label_tran10_2.setFrameShape(QFrame.NoFrame)
        self.label_tran10_2.setFrameShadow(QFrame.Sunken)
        self.label_tran10_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_14.addWidget(self.label_tran10_2)


        self.horizontalLayout_46.addLayout(self.verticalLayout_14)


        self.verticalLayout_9.addLayout(self.horizontalLayout_46)

        self.label_33 = QLabel(self.scrollAreaWidgetContents)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setMaximumSize(QSize(16777215, 10))
        self.label_33.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_33.setFrameShape(QFrame.HLine)

        self.verticalLayout_9.addWidget(self.label_33)

        self.horizontalLayout_53 = QHBoxLayout()
        self.horizontalLayout_53.setObjectName(u"horizontalLayout_53")
        self.verticalLayout_15 = QVBoxLayout()
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.horizontalLayout_60 = QHBoxLayout()
        self.horizontalLayout_60.setObjectName(u"horizontalLayout_60")
        self.verticalLayout_21 = QVBoxLayout()
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.horizontalLayout_69 = QHBoxLayout()
        self.horizontalLayout_69.setObjectName(u"horizontalLayout_69")
        self.label_op_snr_6 = QLabel(self.scrollAreaWidgetContents)
        self.label_op_snr_6.setObjectName(u"label_op_snr_6")
        self.label_op_snr_6.setMaximumSize(QSize(100, 50))
        self.label_op_snr_6.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_6.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_69.addWidget(self.label_op_snr_6)


        self.verticalLayout_21.addLayout(self.horizontalLayout_69)

        self.horizontalLayout_68 = QHBoxLayout()
        self.horizontalLayout_68.setObjectName(u"horizontalLayout_68")
        self.comboBox_rl_datachoice = QComboBox(self.scrollAreaWidgetContents)
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.addItem("")
        self.comboBox_rl_datachoice.setObjectName(u"comboBox_rl_datachoice")
        self.comboBox_rl_datachoice.setMinimumSize(QSize(120, 40))
        self.comboBox_rl_datachoice.setMaximumSize(QSize(100, 16777215))
        self.comboBox_rl_datachoice.setFont(font4)
        self.comboBox_rl_datachoice.setAutoFillBackground(False)
        self.comboBox_rl_datachoice.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_rl_datachoice.setInsertPolicy(QComboBox.InsertBeforeCurrent)
        self.comboBox_rl_datachoice.setIconSize(QSize(16, 16))
        self.comboBox_rl_datachoice.setFrame(True)

        self.horizontalLayout_68.addWidget(self.comboBox_rl_datachoice)


        self.verticalLayout_21.addLayout(self.horizontalLayout_68)


        self.horizontalLayout_60.addLayout(self.verticalLayout_21)

        self.verticalLayout_22 = QVBoxLayout()
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.horizontalLayout_71 = QHBoxLayout()
        self.horizontalLayout_71.setObjectName(u"horizontalLayout_71")
        self.label_op_snr_8 = QLabel(self.scrollAreaWidgetContents)
        self.label_op_snr_8.setObjectName(u"label_op_snr_8")
        self.label_op_snr_8.setMaximumSize(QSize(140, 50))
        self.label_op_snr_8.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_8.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_71.addWidget(self.label_op_snr_8)


        self.verticalLayout_22.addLayout(self.horizontalLayout_71)

        self.horizontalLayout_70 = QHBoxLayout()
        self.horizontalLayout_70.setObjectName(u"horizontalLayout_70")
        self.label_rl_real = QLabel(self.scrollAreaWidgetContents)
        self.label_rl_real.setObjectName(u"label_rl_real")
        self.label_rl_real.setMaximumSize(QSize(140, 50))
        self.label_rl_real.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_rl_real.setFrameShape(QFrame.NoFrame)
        self.label_rl_real.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_70.addWidget(self.label_rl_real)


        self.verticalLayout_22.addLayout(self.horizontalLayout_70)


        self.horizontalLayout_60.addLayout(self.verticalLayout_22)

        self.verticalLayout_23 = QVBoxLayout()
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.horizontalLayout_66 = QHBoxLayout()
        self.horizontalLayout_66.setObjectName(u"horizontalLayout_66")
        self.label_op_snr_7 = QLabel(self.scrollAreaWidgetContents)
        self.label_op_snr_7.setObjectName(u"label_op_snr_7")
        self.label_op_snr_7.setMaximumSize(QSize(140, 50))
        self.label_op_snr_7.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_7.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_66.addWidget(self.label_op_snr_7)


        self.verticalLayout_23.addLayout(self.horizontalLayout_66)

        self.horizontalLayout_67 = QHBoxLayout()
        self.horizontalLayout_67.setObjectName(u"horizontalLayout_67")
        self.label_rl_rec = QLabel(self.scrollAreaWidgetContents)
        self.label_rl_rec.setObjectName(u"label_rl_rec")
        self.label_rl_rec.setMaximumSize(QSize(140, 50))
        self.label_rl_rec.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_rl_rec.setFrameShape(QFrame.NoFrame)
        self.label_rl_rec.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_67.addWidget(self.label_rl_rec)


        self.verticalLayout_23.addLayout(self.horizontalLayout_67)


        self.horizontalLayout_60.addLayout(self.verticalLayout_23)

        self.verticalLayout_27 = QVBoxLayout()
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.label_op_snr_9 = QLabel(self.scrollAreaWidgetContents)
        self.label_op_snr_9.setObjectName(u"label_op_snr_9")
        self.label_op_snr_9.setMaximumSize(QSize(140, 50))
        self.label_op_snr_9.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_9.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.verticalLayout_27.addWidget(self.label_op_snr_9)

        self.layout_rl_next = QHBoxLayout()
        self.layout_rl_next.setObjectName(u"layout_rl_next")

        self.verticalLayout_27.addLayout(self.layout_rl_next)


        self.horizontalLayout_60.addLayout(self.verticalLayout_27)


        self.verticalLayout_15.addLayout(self.horizontalLayout_60)

        self.verticalLayout_24 = QVBoxLayout()
        self.verticalLayout_24.setSpacing(0)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_25 = QVBoxLayout()
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.pic_rl_data_real = QLabel(self.scrollAreaWidgetContents)
        self.pic_rl_data_real.setObjectName(u"pic_rl_data_real")
        self.pic_rl_data_real.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_rl_data_real.setAlignment(Qt.AlignCenter)

        self.verticalLayout_25.addWidget(self.pic_rl_data_real)


        self.verticalLayout_24.addLayout(self.verticalLayout_25)

        self.verticalLayout_26 = QVBoxLayout()
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.pic_rl_data_virtual = QLabel(self.scrollAreaWidgetContents)
        self.pic_rl_data_virtual.setObjectName(u"pic_rl_data_virtual")
        self.pic_rl_data_virtual.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_rl_data_virtual.setAlignment(Qt.AlignCenter)

        self.verticalLayout_26.addWidget(self.pic_rl_data_virtual)


        self.verticalLayout_24.addLayout(self.verticalLayout_26)


        self.verticalLayout_15.addLayout(self.verticalLayout_24)

        self.label_tran10_4 = QLabel(self.scrollAreaWidgetContents)
        self.label_tran10_4.setObjectName(u"label_tran10_4")
        self.label_tran10_4.setMaximumSize(QSize(16777215, 30))
        self.label_tran10_4.setFrameShape(QFrame.NoFrame)
        self.label_tran10_4.setFrameShadow(QFrame.Sunken)
        self.label_tran10_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_15.addWidget(self.label_tran10_4)


        self.horizontalLayout_53.addLayout(self.verticalLayout_15)

        self.verticalLayout_20 = QVBoxLayout()
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.pic_rl_shipin = QLabel(self.scrollAreaWidgetContents)
        self.pic_rl_shipin.setObjectName(u"pic_rl_shipin")
        self.pic_rl_shipin.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_rl_shipin.setAlignment(Qt.AlignCenter)

        self.verticalLayout_20.addWidget(self.pic_rl_shipin)

        self.label_tran10_3 = QLabel(self.scrollAreaWidgetContents)
        self.label_tran10_3.setObjectName(u"label_tran10_3")
        self.label_tran10_3.setMaximumSize(QSize(16777215, 30))
        self.label_tran10_3.setFrameShape(QFrame.NoFrame)
        self.label_tran10_3.setFrameShadow(QFrame.Sunken)
        self.label_tran10_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_20.addWidget(self.label_tran10_3)


        self.horizontalLayout_53.addLayout(self.verticalLayout_20)


        self.verticalLayout_9.addLayout(self.horizontalLayout_53)

        self.scroll_area_rl.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_6.addWidget(self.scroll_area_rl)

        self.pages.addWidget(self.page_RL)
        self.page_IN = QWidget()
        self.page_IN.setObjectName(u"page_IN")
        self.verticalLayout_17 = QVBoxLayout(self.page_IN)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.label_62 = QLabel(self.page_IN)
        self.label_62.setObjectName(u"label_62")
        self.label_62.setMaximumSize(QSize(16777215, 10))

        self.verticalLayout_17.addWidget(self.label_62)

        self.title_label_op_4 = QLabel(self.page_IN)
        self.title_label_op_4.setObjectName(u"title_label_op_4")
        self.title_label_op_4.setMaximumSize(QSize(16777215, 40))
        self.title_label_op_4.setFont(font3)
        self.title_label_op_4.setStyleSheet(u"font: 26pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.title_label_op_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_17.addWidget(self.title_label_op_4)

        self.label_39 = QLabel(self.page_IN)
        self.label_39.setObjectName(u"label_39")
        self.label_39.setMaximumSize(QSize(16777215, 10))

        self.verticalLayout_17.addWidget(self.label_39)

        self.horizontalLayout_82 = QHBoxLayout()
        self.horizontalLayout_82.setObjectName(u"horizontalLayout_82")
        self.label_41 = QLabel(self.page_IN)
        self.label_41.setObjectName(u"label_41")

        self.horizontalLayout_82.addWidget(self.label_41)

        self.horizontalLayout_83 = QHBoxLayout()
        self.horizontalLayout_83.setObjectName(u"horizontalLayout_83")
        self.label_op_snr_11 = QLabel(self.page_IN)
        self.label_op_snr_11.setObjectName(u"label_op_snr_11")
        self.label_op_snr_11.setMaximumSize(QSize(16777215, 50))
        self.label_op_snr_11.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_11.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_83.addWidget(self.label_op_snr_11)


        self.horizontalLayout_82.addLayout(self.horizontalLayout_83)

        self.horizontalLayout_84 = QHBoxLayout()
        self.horizontalLayout_84.setObjectName(u"horizontalLayout_84")
        self.comboBox_in_dataset = QComboBox(self.page_IN)
        self.comboBox_in_dataset.addItem("")
        self.comboBox_in_dataset.setObjectName(u"comboBox_in_dataset")
        self.comboBox_in_dataset.setMinimumSize(QSize(150, 40))
        self.comboBox_in_dataset.setMaximumSize(QSize(120, 50))
        self.comboBox_in_dataset.setFont(font4)
        self.comboBox_in_dataset.setAutoFillBackground(False)
        self.comboBox_in_dataset.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_in_dataset.setIconSize(QSize(16, 16))
        self.comboBox_in_dataset.setFrame(True)

        self.horizontalLayout_84.addWidget(self.comboBox_in_dataset)


        self.horizontalLayout_82.addLayout(self.horizontalLayout_84)

        self.label_42 = QLabel(self.page_IN)
        self.label_42.setObjectName(u"label_42")

        self.horizontalLayout_82.addWidget(self.label_42)

        self.horizontalLayout_54 = QHBoxLayout()
        self.horizontalLayout_54.setObjectName(u"horizontalLayout_54")
        self.label_op_snr_10 = QLabel(self.page_IN)
        self.label_op_snr_10.setObjectName(u"label_op_snr_10")
        self.label_op_snr_10.setMaximumSize(QSize(16777215, 50))
        self.label_op_snr_10.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_snr_10.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_54.addWidget(self.label_op_snr_10)


        self.horizontalLayout_82.addLayout(self.horizontalLayout_54)

        self.horizontalLayout_86 = QHBoxLayout()
        self.horizontalLayout_86.setObjectName(u"horizontalLayout_86")
        self.comboBox_in_snr = QComboBox(self.page_IN)
        self.comboBox_in_snr.addItem("")
        self.comboBox_in_snr.setObjectName(u"comboBox_in_snr")
        self.comboBox_in_snr.setMinimumSize(QSize(80, 40))
        self.comboBox_in_snr.setMaximumSize(QSize(120, 50))
        self.comboBox_in_snr.setFont(font4)
        self.comboBox_in_snr.setAutoFillBackground(False)
        self.comboBox_in_snr.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_in_snr.setIconSize(QSize(16, 16))
        self.comboBox_in_snr.setFrame(True)

        self.horizontalLayout_86.addWidget(self.comboBox_in_snr)


        self.horizontalLayout_82.addLayout(self.horizontalLayout_86)

        self.label_43 = QLabel(self.page_IN)
        self.label_43.setObjectName(u"label_43")

        self.horizontalLayout_82.addWidget(self.label_43)

        self.horizontalLayout_85 = QHBoxLayout()
        self.horizontalLayout_85.setObjectName(u"horizontalLayout_85")
        self.label_op_data_3 = QLabel(self.page_IN)
        self.label_op_data_3.setObjectName(u"label_op_data_3")
        self.label_op_data_3.setMaximumSize(QSize(16777215, 50))
        self.label_op_data_3.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_data_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_85.addWidget(self.label_op_data_3)


        self.horizontalLayout_82.addLayout(self.horizontalLayout_85)

        self.horizontalLayout_87 = QHBoxLayout()
        self.horizontalLayout_87.setObjectName(u"horizontalLayout_87")
        self.comboBox_in_method = QComboBox(self.page_IN)
        self.comboBox_in_method.addItem("")
        self.comboBox_in_method.setObjectName(u"comboBox_in_method")
        self.comboBox_in_method.setMinimumSize(QSize(180, 40))
        self.comboBox_in_method.setMaximumSize(QSize(100, 50))
        self.comboBox_in_method.setFont(font4)
        self.comboBox_in_method.setAutoFillBackground(False)
        self.comboBox_in_method.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_in_method.setIconSize(QSize(16, 16))
        self.comboBox_in_method.setFrame(True)

        self.horizontalLayout_87.addWidget(self.comboBox_in_method)


        self.horizontalLayout_82.addLayout(self.horizontalLayout_87)

        self.label_44 = QLabel(self.page_IN)
        self.label_44.setObjectName(u"label_44")

        self.horizontalLayout_82.addWidget(self.label_44)

        self.horizontalLayout_59 = QHBoxLayout()
        self.horizontalLayout_59.setObjectName(u"horizontalLayout_59")
        self.label_op_data_4 = QLabel(self.page_IN)
        self.label_op_data_4.setObjectName(u"label_op_data_4")
        self.label_op_data_4.setMaximumSize(QSize(16777215, 50))
        self.label_op_data_4.setStyleSheet(u"font: 16pt \"\u5fae\u8f6f\u96c5\u9ed1\";")
        self.label_op_data_4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_59.addWidget(self.label_op_data_4)


        self.horizontalLayout_82.addLayout(self.horizontalLayout_59)

        self.horizontalLayout_58 = QHBoxLayout()
        self.horizontalLayout_58.setObjectName(u"horizontalLayout_58")
        self.comboBox_in_layer = QComboBox(self.page_IN)
        self.comboBox_in_layer.addItem("")
        self.comboBox_in_layer.addItem("")
        self.comboBox_in_layer.addItem("")
        self.comboBox_in_layer.addItem("")
        self.comboBox_in_layer.setObjectName(u"comboBox_in_layer")
        self.comboBox_in_layer.setMinimumSize(QSize(100, 40))
        self.comboBox_in_layer.setMaximumSize(QSize(200, 50))
        self.comboBox_in_layer.setFont(font4)
        self.comboBox_in_layer.setAutoFillBackground(False)
        self.comboBox_in_layer.setStyleSheet(u"background-color: rgb(27, 29, 35);\n"
"border-radius: 5px;\n"
"border: 2px solid rgb(33, 37, 43);\n"
"padding: 5px;\n"
"padding-left: 10px;\n"
"font: 14pt \"Segoe UI\";")
        self.comboBox_in_layer.setIconSize(QSize(16, 16))
        self.comboBox_in_layer.setFrame(True)

        self.horizontalLayout_58.addWidget(self.comboBox_in_layer)


        self.horizontalLayout_82.addLayout(self.horizontalLayout_58)

        self.label_56 = QLabel(self.page_IN)
        self.label_56.setObjectName(u"label_56")

        self.horizontalLayout_82.addWidget(self.label_56)

        self.layout_in_next = QHBoxLayout()
        self.layout_in_next.setObjectName(u"layout_in_next")

        self.horizontalLayout_82.addLayout(self.layout_in_next)

        self.label_40 = QLabel(self.page_IN)
        self.label_40.setObjectName(u"label_40")

        self.horizontalLayout_82.addWidget(self.label_40)


        self.verticalLayout_17.addLayout(self.horizontalLayout_82)

        self.label_45 = QLabel(self.page_IN)
        self.label_45.setObjectName(u"label_45")
        self.label_45.setMaximumSize(QSize(16777215, 10))
        self.label_45.setStyleSheet(u"border-color: rgb(27, 30, 35);")
        self.label_45.setFrameShape(QFrame.HLine)

        self.verticalLayout_17.addWidget(self.label_45)

        self.scrollArea_2 = QScrollArea(self.page_IN)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setStyleSheet(u"background: transparent;")
        self.scrollArea_2.setFrameShape(QFrame.NoFrame)
        self.scrollArea_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 1272, 912))
        self.scrollAreaWidgetContents_2.setStyleSheet(u"background: transparent;")
        self.verticalLayout_18 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.horizontalLayout_81 = QHBoxLayout()
        self.horizontalLayout_81.setObjectName(u"horizontalLayout_81")
        self.label_in_mod_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_1.setObjectName(u"label_in_mod_1")
        self.label_in_mod_1.setMaximumSize(QSize(100, 30))
        self.label_in_mod_1.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_1.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_81.addWidget(self.label_in_mod_1)

        self.horizontalLayout_93 = QHBoxLayout()
        self.horizontalLayout_93.setObjectName(u"horizontalLayout_93")
        self.pic_in_mod1_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod1_1.setObjectName(u"pic_in_mod1_1")
        self.pic_in_mod1_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod1_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_93.addWidget(self.pic_in_mod1_1)


        self.horizontalLayout_81.addLayout(self.horizontalLayout_93)

        self.horizontalLayout_92 = QHBoxLayout()
        self.horizontalLayout_92.setObjectName(u"horizontalLayout_92")
        self.pic_in_mod1_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod1_2.setObjectName(u"pic_in_mod1_2")
        self.pic_in_mod1_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod1_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_92.addWidget(self.pic_in_mod1_2)


        self.horizontalLayout_81.addLayout(self.horizontalLayout_92)

        self.horizontalLayout_91 = QHBoxLayout()
        self.horizontalLayout_91.setObjectName(u"horizontalLayout_91")
        self.pic_in_mod1_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod1_3.setObjectName(u"pic_in_mod1_3")
        self.pic_in_mod1_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod1_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_91.addWidget(self.pic_in_mod1_3)


        self.horizontalLayout_81.addLayout(self.horizontalLayout_91)


        self.verticalLayout_18.addLayout(self.horizontalLayout_81)

        self.label_46 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_46.setObjectName(u"label_46")
        self.label_46.setMaximumSize(QSize(16777215, 10))
        self.label_46.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_46.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_46)

        self.horizontalLayout_94 = QHBoxLayout()
        self.horizontalLayout_94.setObjectName(u"horizontalLayout_94")
        self.label_in_mod_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_2.setObjectName(u"label_in_mod_2")
        self.label_in_mod_2.setMaximumSize(QSize(100, 30))
        self.label_in_mod_2.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_2.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_94.addWidget(self.label_in_mod_2)

        self.horizontalLayout_95 = QHBoxLayout()
        self.horizontalLayout_95.setObjectName(u"horizontalLayout_95")
        self.pic_in_mod2_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod2_1.setObjectName(u"pic_in_mod2_1")
        self.pic_in_mod2_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod2_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_95.addWidget(self.pic_in_mod2_1)


        self.horizontalLayout_94.addLayout(self.horizontalLayout_95)

        self.horizontalLayout_96 = QHBoxLayout()
        self.horizontalLayout_96.setObjectName(u"horizontalLayout_96")
        self.pic_in_mod2_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod2_2.setObjectName(u"pic_in_mod2_2")
        self.pic_in_mod2_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod2_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_96.addWidget(self.pic_in_mod2_2)


        self.horizontalLayout_94.addLayout(self.horizontalLayout_96)

        self.horizontalLayout_97 = QHBoxLayout()
        self.horizontalLayout_97.setObjectName(u"horizontalLayout_97")
        self.pic_in_mod2_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod2_3.setObjectName(u"pic_in_mod2_3")
        self.pic_in_mod2_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod2_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_97.addWidget(self.pic_in_mod2_3)


        self.horizontalLayout_94.addLayout(self.horizontalLayout_97)


        self.verticalLayout_18.addLayout(self.horizontalLayout_94)

        self.label_47 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_47.setObjectName(u"label_47")
        self.label_47.setMaximumSize(QSize(16777215, 10))
        self.label_47.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_47.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_47)

        self.horizontalLayout_99 = QHBoxLayout()
        self.horizontalLayout_99.setObjectName(u"horizontalLayout_99")
        self.label_in_mod_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_3.setObjectName(u"label_in_mod_3")
        self.label_in_mod_3.setMaximumSize(QSize(100, 30))
        self.label_in_mod_3.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_3.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_99.addWidget(self.label_in_mod_3)

        self.horizontalLayout_100 = QHBoxLayout()
        self.horizontalLayout_100.setObjectName(u"horizontalLayout_100")
        self.pic_in_mod3_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod3_1.setObjectName(u"pic_in_mod3_1")
        self.pic_in_mod3_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod3_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_100.addWidget(self.pic_in_mod3_1)


        self.horizontalLayout_99.addLayout(self.horizontalLayout_100)

        self.horizontalLayout_101 = QHBoxLayout()
        self.horizontalLayout_101.setObjectName(u"horizontalLayout_101")
        self.pic_in_mod3_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod3_2.setObjectName(u"pic_in_mod3_2")
        self.pic_in_mod3_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod3_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_101.addWidget(self.pic_in_mod3_2)


        self.horizontalLayout_99.addLayout(self.horizontalLayout_101)

        self.horizontalLayout_102 = QHBoxLayout()
        self.horizontalLayout_102.setObjectName(u"horizontalLayout_102")
        self.pic_in_mod3_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod3_3.setObjectName(u"pic_in_mod3_3")
        self.pic_in_mod3_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod3_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_102.addWidget(self.pic_in_mod3_3)


        self.horizontalLayout_99.addLayout(self.horizontalLayout_102)


        self.verticalLayout_18.addLayout(self.horizontalLayout_99)

        self.label_48 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_48.setObjectName(u"label_48")
        self.label_48.setMaximumSize(QSize(16777215, 10))
        self.label_48.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_48.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_48)

        self.horizontalLayout_104 = QHBoxLayout()
        self.horizontalLayout_104.setObjectName(u"horizontalLayout_104")
        self.label_in_mod_4 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_4.setObjectName(u"label_in_mod_4")
        self.label_in_mod_4.setMaximumSize(QSize(100, 30))
        self.label_in_mod_4.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_4.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_4.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_104.addWidget(self.label_in_mod_4)

        self.horizontalLayout_105 = QHBoxLayout()
        self.horizontalLayout_105.setObjectName(u"horizontalLayout_105")
        self.pic_in_mod4_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod4_1.setObjectName(u"pic_in_mod4_1")
        self.pic_in_mod4_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod4_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_105.addWidget(self.pic_in_mod4_1)


        self.horizontalLayout_104.addLayout(self.horizontalLayout_105)

        self.horizontalLayout_106 = QHBoxLayout()
        self.horizontalLayout_106.setObjectName(u"horizontalLayout_106")
        self.pic_in_mod4_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod4_2.setObjectName(u"pic_in_mod4_2")
        self.pic_in_mod4_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod4_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_106.addWidget(self.pic_in_mod4_2)


        self.horizontalLayout_104.addLayout(self.horizontalLayout_106)

        self.horizontalLayout_107 = QHBoxLayout()
        self.horizontalLayout_107.setObjectName(u"horizontalLayout_107")
        self.pic_in_mod4_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod4_3.setObjectName(u"pic_in_mod4_3")
        self.pic_in_mod4_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod4_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_107.addWidget(self.pic_in_mod4_3)


        self.horizontalLayout_104.addLayout(self.horizontalLayout_107)


        self.verticalLayout_18.addLayout(self.horizontalLayout_104)

        self.label_49 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_49.setObjectName(u"label_49")
        self.label_49.setMaximumSize(QSize(16777215, 10))
        self.label_49.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_49.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_49)

        self.horizontalLayout_109 = QHBoxLayout()
        self.horizontalLayout_109.setObjectName(u"horizontalLayout_109")
        self.label_in_mod_5 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_5.setObjectName(u"label_in_mod_5")
        self.label_in_mod_5.setMaximumSize(QSize(100, 30))
        self.label_in_mod_5.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_5.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_5.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_109.addWidget(self.label_in_mod_5)

        self.horizontalLayout_110 = QHBoxLayout()
        self.horizontalLayout_110.setObjectName(u"horizontalLayout_110")
        self.pic_in_mod5_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod5_1.setObjectName(u"pic_in_mod5_1")
        self.pic_in_mod5_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod5_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_110.addWidget(self.pic_in_mod5_1)


        self.horizontalLayout_109.addLayout(self.horizontalLayout_110)

        self.horizontalLayout_111 = QHBoxLayout()
        self.horizontalLayout_111.setObjectName(u"horizontalLayout_111")
        self.pic_in_mod5_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod5_2.setObjectName(u"pic_in_mod5_2")
        self.pic_in_mod5_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod5_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_111.addWidget(self.pic_in_mod5_2)


        self.horizontalLayout_109.addLayout(self.horizontalLayout_111)

        self.horizontalLayout_112 = QHBoxLayout()
        self.horizontalLayout_112.setObjectName(u"horizontalLayout_112")
        self.pic_in_mod5_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod5_3.setObjectName(u"pic_in_mod5_3")
        self.pic_in_mod5_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod5_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_112.addWidget(self.pic_in_mod5_3)


        self.horizontalLayout_109.addLayout(self.horizontalLayout_112)


        self.verticalLayout_18.addLayout(self.horizontalLayout_109)

        self.label_50 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_50.setObjectName(u"label_50")
        self.label_50.setMaximumSize(QSize(16777215, 10))
        self.label_50.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_50.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_50)

        self.horizontalLayout_114 = QHBoxLayout()
        self.horizontalLayout_114.setObjectName(u"horizontalLayout_114")
        self.label_in_mod_6 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_6.setObjectName(u"label_in_mod_6")
        self.label_in_mod_6.setMaximumSize(QSize(100, 30))
        self.label_in_mod_6.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_6.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_6.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_114.addWidget(self.label_in_mod_6)

        self.horizontalLayout_115 = QHBoxLayout()
        self.horizontalLayout_115.setObjectName(u"horizontalLayout_115")
        self.pic_in_mod6_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod6_1.setObjectName(u"pic_in_mod6_1")
        self.pic_in_mod6_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod6_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_115.addWidget(self.pic_in_mod6_1)


        self.horizontalLayout_114.addLayout(self.horizontalLayout_115)

        self.horizontalLayout_116 = QHBoxLayout()
        self.horizontalLayout_116.setObjectName(u"horizontalLayout_116")
        self.pic_in_mod6_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod6_2.setObjectName(u"pic_in_mod6_2")
        self.pic_in_mod6_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod6_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_116.addWidget(self.pic_in_mod6_2)


        self.horizontalLayout_114.addLayout(self.horizontalLayout_116)

        self.horizontalLayout_117 = QHBoxLayout()
        self.horizontalLayout_117.setObjectName(u"horizontalLayout_117")
        self.pic_in_mod6_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod6_3.setObjectName(u"pic_in_mod6_3")
        self.pic_in_mod6_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod6_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_117.addWidget(self.pic_in_mod6_3)


        self.horizontalLayout_114.addLayout(self.horizontalLayout_117)


        self.verticalLayout_18.addLayout(self.horizontalLayout_114)

        self.label_51 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_51.setObjectName(u"label_51")
        self.label_51.setMaximumSize(QSize(16777215, 10))
        self.label_51.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_51.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_51)

        self.horizontalLayout_119 = QHBoxLayout()
        self.horizontalLayout_119.setObjectName(u"horizontalLayout_119")
        self.label_in_mod_7 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_7.setObjectName(u"label_in_mod_7")
        self.label_in_mod_7.setMaximumSize(QSize(100, 30))
        self.label_in_mod_7.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_7.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_7.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_119.addWidget(self.label_in_mod_7)

        self.horizontalLayout_120 = QHBoxLayout()
        self.horizontalLayout_120.setObjectName(u"horizontalLayout_120")
        self.pic_in_mod7_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod7_1.setObjectName(u"pic_in_mod7_1")
        self.pic_in_mod7_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod7_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_120.addWidget(self.pic_in_mod7_1)


        self.horizontalLayout_119.addLayout(self.horizontalLayout_120)

        self.horizontalLayout_121 = QHBoxLayout()
        self.horizontalLayout_121.setObjectName(u"horizontalLayout_121")
        self.pic_in_mod7_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod7_2.setObjectName(u"pic_in_mod7_2")
        self.pic_in_mod7_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod7_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_121.addWidget(self.pic_in_mod7_2)


        self.horizontalLayout_119.addLayout(self.horizontalLayout_121)

        self.horizontalLayout_122 = QHBoxLayout()
        self.horizontalLayout_122.setObjectName(u"horizontalLayout_122")
        self.pic_in_mod7_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod7_3.setObjectName(u"pic_in_mod7_3")
        self.pic_in_mod7_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod7_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_122.addWidget(self.pic_in_mod7_3)


        self.horizontalLayout_119.addLayout(self.horizontalLayout_122)


        self.verticalLayout_18.addLayout(self.horizontalLayout_119)

        self.label_52 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_52.setObjectName(u"label_52")
        self.label_52.setMaximumSize(QSize(16777215, 10))
        self.label_52.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_52.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_52)

        self.horizontalLayout_124 = QHBoxLayout()
        self.horizontalLayout_124.setObjectName(u"horizontalLayout_124")
        self.label_in_mod_8 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_8.setObjectName(u"label_in_mod_8")
        self.label_in_mod_8.setMaximumSize(QSize(100, 30))
        self.label_in_mod_8.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_8.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_8.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_124.addWidget(self.label_in_mod_8)

        self.horizontalLayout_125 = QHBoxLayout()
        self.horizontalLayout_125.setObjectName(u"horizontalLayout_125")
        self.pic_in_mod8_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod8_1.setObjectName(u"pic_in_mod8_1")
        self.pic_in_mod8_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod8_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_125.addWidget(self.pic_in_mod8_1)


        self.horizontalLayout_124.addLayout(self.horizontalLayout_125)

        self.horizontalLayout_126 = QHBoxLayout()
        self.horizontalLayout_126.setObjectName(u"horizontalLayout_126")
        self.pic_in_mod8_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod8_2.setObjectName(u"pic_in_mod8_2")
        self.pic_in_mod8_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod8_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_126.addWidget(self.pic_in_mod8_2)


        self.horizontalLayout_124.addLayout(self.horizontalLayout_126)

        self.horizontalLayout_127 = QHBoxLayout()
        self.horizontalLayout_127.setObjectName(u"horizontalLayout_127")
        self.pic_in_mod8_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod8_3.setObjectName(u"pic_in_mod8_3")
        self.pic_in_mod8_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod8_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_127.addWidget(self.pic_in_mod8_3)


        self.horizontalLayout_124.addLayout(self.horizontalLayout_127)


        self.verticalLayout_18.addLayout(self.horizontalLayout_124)

        self.label_53 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_53.setObjectName(u"label_53")
        self.label_53.setMaximumSize(QSize(16777215, 10))
        self.label_53.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_53.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_53)

        self.horizontalLayout_129 = QHBoxLayout()
        self.horizontalLayout_129.setObjectName(u"horizontalLayout_129")
        self.label_in_mod_9 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_9.setObjectName(u"label_in_mod_9")
        self.label_in_mod_9.setMaximumSize(QSize(100, 30))
        self.label_in_mod_9.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_9.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_9.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_129.addWidget(self.label_in_mod_9)

        self.horizontalLayout_130 = QHBoxLayout()
        self.horizontalLayout_130.setObjectName(u"horizontalLayout_130")
        self.pic_in_mod9_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod9_1.setObjectName(u"pic_in_mod9_1")
        self.pic_in_mod9_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod9_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_130.addWidget(self.pic_in_mod9_1)


        self.horizontalLayout_129.addLayout(self.horizontalLayout_130)

        self.horizontalLayout_131 = QHBoxLayout()
        self.horizontalLayout_131.setObjectName(u"horizontalLayout_131")
        self.pic_in_mod9_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod9_2.setObjectName(u"pic_in_mod9_2")
        self.pic_in_mod9_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod9_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_131.addWidget(self.pic_in_mod9_2)


        self.horizontalLayout_129.addLayout(self.horizontalLayout_131)

        self.horizontalLayout_132 = QHBoxLayout()
        self.horizontalLayout_132.setObjectName(u"horizontalLayout_132")
        self.pic_in_mod9_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod9_3.setObjectName(u"pic_in_mod9_3")
        self.pic_in_mod9_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod9_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_132.addWidget(self.pic_in_mod9_3)


        self.horizontalLayout_129.addLayout(self.horizontalLayout_132)


        self.verticalLayout_18.addLayout(self.horizontalLayout_129)

        self.label_54 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_54.setObjectName(u"label_54")
        self.label_54.setMaximumSize(QSize(16777215, 10))
        self.label_54.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_54.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_54)

        self.horizontalLayout_134 = QHBoxLayout()
        self.horizontalLayout_134.setObjectName(u"horizontalLayout_134")
        self.label_in_mod_10 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_10.setObjectName(u"label_in_mod_10")
        self.label_in_mod_10.setMaximumSize(QSize(100, 30))
        self.label_in_mod_10.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_10.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_10.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_134.addWidget(self.label_in_mod_10)

        self.horizontalLayout_135 = QHBoxLayout()
        self.horizontalLayout_135.setObjectName(u"horizontalLayout_135")
        self.pic_in_mod10_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod10_1.setObjectName(u"pic_in_mod10_1")
        self.pic_in_mod10_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod10_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_135.addWidget(self.pic_in_mod10_1)


        self.horizontalLayout_134.addLayout(self.horizontalLayout_135)

        self.horizontalLayout_136 = QHBoxLayout()
        self.horizontalLayout_136.setObjectName(u"horizontalLayout_136")
        self.pic_in_mod10_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod10_2.setObjectName(u"pic_in_mod10_2")
        self.pic_in_mod10_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod10_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_136.addWidget(self.pic_in_mod10_2)


        self.horizontalLayout_134.addLayout(self.horizontalLayout_136)

        self.horizontalLayout_137 = QHBoxLayout()
        self.horizontalLayout_137.setObjectName(u"horizontalLayout_137")
        self.pic_in_mod10_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod10_3.setObjectName(u"pic_in_mod10_3")
        self.pic_in_mod10_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod10_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_137.addWidget(self.pic_in_mod10_3)


        self.horizontalLayout_134.addLayout(self.horizontalLayout_137)


        self.verticalLayout_18.addLayout(self.horizontalLayout_134)

        self.label_55 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_55.setObjectName(u"label_55")
        self.label_55.setMaximumSize(QSize(16777215, 10))
        self.label_55.setStyleSheet(u"color: rgb(27, 30, 35);")
        self.label_55.setFrameShape(QFrame.HLine)

        self.verticalLayout_18.addWidget(self.label_55)

        self.horizontalLayout_139 = QHBoxLayout()
        self.horizontalLayout_139.setObjectName(u"horizontalLayout_139")
        self.label_in_mod_11 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_in_mod_11.setObjectName(u"label_in_mod_11")
        self.label_in_mod_11.setMaximumSize(QSize(100, 30))
        self.label_in_mod_11.setFrameShape(QFrame.NoFrame)
        self.label_in_mod_11.setFrameShadow(QFrame.Sunken)
        self.label_in_mod_11.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_139.addWidget(self.label_in_mod_11)

        self.horizontalLayout_140 = QHBoxLayout()
        self.horizontalLayout_140.setObjectName(u"horizontalLayout_140")
        self.pic_in_mod11_1 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod11_1.setObjectName(u"pic_in_mod11_1")
        self.pic_in_mod11_1.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod11_1.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_140.addWidget(self.pic_in_mod11_1)


        self.horizontalLayout_139.addLayout(self.horizontalLayout_140)

        self.horizontalLayout_141 = QHBoxLayout()
        self.horizontalLayout_141.setObjectName(u"horizontalLayout_141")
        self.pic_in_mod11_2 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod11_2.setObjectName(u"pic_in_mod11_2")
        self.pic_in_mod11_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod11_2.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_141.addWidget(self.pic_in_mod11_2)


        self.horizontalLayout_139.addLayout(self.horizontalLayout_141)

        self.horizontalLayout_142 = QHBoxLayout()
        self.horizontalLayout_142.setObjectName(u"horizontalLayout_142")
        self.pic_in_mod11_3 = QLabel(self.scrollAreaWidgetContents_2)
        self.pic_in_mod11_3.setObjectName(u"pic_in_mod11_3")
        self.pic_in_mod11_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.pic_in_mod11_3.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_142.addWidget(self.pic_in_mod11_3)


        self.horizontalLayout_139.addLayout(self.horizontalLayout_142)


        self.verticalLayout_18.addLayout(self.horizontalLayout_139)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_17.addWidget(self.scrollArea_2)

        self.pages.addWidget(self.page_IN)

        self.main_pages_layout.addWidget(self.pages)


        self.retranslateUi(MainPages)

        self.pages.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(MainPages)
    # setupUi

    def retranslateUi(self, MainPages):
        MainPages.setWindowTitle(QCoreApplication.translate("MainPages", u"Form", None))
        self.label.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-family:'\u5fae\u8f6f\u96c5\u9ed1'; font-size:28pt; color:#6c99f4;\">\u7535\u78c1\u4fe1\u53f7\u76ee\u6807\u8bc6\u522b\u6280\u672f\u539f\u7406\u9a8c\u8bc1\u7cfb\u7edf</span></p></body></html>", None))
        self.title_label.setText(QCoreApplication.translate("MainPages", u"Custom Widgets Page", None))
        self.description_label.setText(QCoreApplication.translate("MainPages", u"Here will be all the custom widgets, they will be added over time on this page.\n"
"I will try to always record a new tutorial when adding a new Widget and updating the project on Patreon before launching on GitHub and GitHub after the public release.", None))
        self.title_label_op.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-size:26pt; font-weight:700;\">\u9762\u5411\u7535\u78c1\u4fe1\u53f7\u76ee\u6807\u7279\u5f81\u7684\u7f51\u7edc\u6a21\u578b\u4f18\u5316</span></p></body></html>", None))
        self.label_27.setText("")
        self.label_op_data.setText(QCoreApplication.translate("MainPages", u"\u6570\u636e\u96c6\uff1a", None))
        self.comboBox_op_data.setItemText(0, QCoreApplication.translate("MainPages", u"RML2016.04c", None))

        self.label_2.setText("")
        self.label_op_snr.setText(QCoreApplication.translate("MainPages", u"\u4fe1\u566a\u6bd4\uff1a", None))
        self.comboBox_op_snr.setItemText(0, QCoreApplication.translate("MainPages", u"6dB-SNR", None))

        self.label_3.setText("")
        self.label_op_sample.setText(QCoreApplication.translate("MainPages", u"\u6d4b\u8bd5\u6837\u672c\u9009\u62e9\uff1a", None))
        self.comboBox_op_sample.setItemText(0, QCoreApplication.translate("MainPages", u"1-4055", None))
        self.comboBox_op_sample.setItemText(1, QCoreApplication.translate("MainPages", u"1-500", None))
        self.comboBox_op_sample.setItemText(2, QCoreApplication.translate("MainPages", u"501-1000", None))
        self.comboBox_op_sample.setItemText(3, QCoreApplication.translate("MainPages", u"1001-1500", None))
        self.comboBox_op_sample.setItemText(4, QCoreApplication.translate("MainPages", u"1501-2000", None))
        self.comboBox_op_sample.setItemText(5, QCoreApplication.translate("MainPages", u"2001-2500", None))
        self.comboBox_op_sample.setItemText(6, QCoreApplication.translate("MainPages", u"2501-3000", None))
        self.comboBox_op_sample.setItemText(7, QCoreApplication.translate("MainPages", u"3001-3500", None))
        self.comboBox_op_sample.setItemText(8, QCoreApplication.translate("MainPages", u"3501-4000", None))
        self.comboBox_op_sample.setItemText(9, QCoreApplication.translate("MainPages", u"4001-4055", None))

        self.label_4.setText("")
        self.label_5.setText("")
        self.label_59.setText("")
        self.pic_op_data_1.setText("")
        self.pic_op_data_2.setText("")
        self.pic_op_data_3.setText("")
        self.pic_op_data_4.setText("")
        self.pic_op_data_5.setText("")
        self.pic_op_data_6.setText("")
        self.pic_op_data_7.setText("")
        self.pic_op_data_8.setText("")
        self.label_6.setText("")
        self.label_op_base_model.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p align=\"center\"><span style=\" font-weight:700;\">\u672a\u4f18\u5316</span></p><p align=\"center\"><span style=\" font-weight:700;\">\u6a21\u578b</span></p></body></html>", None))
        self.label_7.setText("")
        self.label_op_shot_base.setText(QCoreApplication.translate("MainPages", u"\u6bcf\u7c7b\u6837\u672c\u6570\uff1a", None))
        self.comboBox_op_shot_base.setItemText(0, QCoreApplication.translate("MainPages", u"10-shot", None))
        self.comboBox_op_shot_base.setItemText(1, QCoreApplication.translate("MainPages", u"20-shot", None))
        self.comboBox_op_shot_base.setItemText(2, QCoreApplication.translate("MainPages", u"30-shot", None))
        self.comboBox_op_shot_base.setItemText(3, QCoreApplication.translate("MainPages", u"40-shot", None))
        self.comboBox_op_shot_base.setItemText(4, QCoreApplication.translate("MainPages", u"50-shot", None))
        self.comboBox_op_shot_base.setItemText(5, QCoreApplication.translate("MainPages", u"60-shot", None))
        self.comboBox_op_shot_base.setItemText(6, QCoreApplication.translate("MainPages", u"70-shot", None))
        self.comboBox_op_shot_base.setItemText(7, QCoreApplication.translate("MainPages", u"80-shot", None))
        self.comboBox_op_shot_base.setItemText(8, QCoreApplication.translate("MainPages", u"90-shot", None))
        self.comboBox_op_shot_base.setItemText(9, QCoreApplication.translate("MainPages", u"100-shot", None))

        self.label_10.setText("")
        self.label_9.setText("")
        self.label_op_adv_model.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-weight:700;\">\u4f18\u5316</span></p><p><span style=\" font-weight:700;\">\u6a21\u578b</span></p></body></html>", None))
        self.label_op_choice_method.setText(QCoreApplication.translate("MainPages", u"\u4f18\u5316\u65b9\u6cd5\uff1a", None))
        self.comboBox_op_method.setItemText(0, QCoreApplication.translate("MainPages", u"Dropout", None))
        self.comboBox_op_method.setItemText(1, QCoreApplication.translate("MainPages", u"InstanceNrom", None))
        self.comboBox_op_method.setItemText(2, QCoreApplication.translate("MainPages", u"GroupNorm", None))

        self.label_op_shot_adv.setText(QCoreApplication.translate("MainPages", u"\u6bcf\u7c7b\u6837\u672c\u6570\uff1a", None))
        self.comboBox_op_shot_adv.setItemText(0, QCoreApplication.translate("MainPages", u"10-shot", None))
        self.comboBox_op_shot_adv.setItemText(1, QCoreApplication.translate("MainPages", u"20-shot", None))
        self.comboBox_op_shot_adv.setItemText(2, QCoreApplication.translate("MainPages", u"30-shot", None))
        self.comboBox_op_shot_adv.setItemText(3, QCoreApplication.translate("MainPages", u"40-shot", None))
        self.comboBox_op_shot_adv.setItemText(4, QCoreApplication.translate("MainPages", u"50-shot", None))
        self.comboBox_op_shot_adv.setItemText(5, QCoreApplication.translate("MainPages", u"60-shot", None))
        self.comboBox_op_shot_adv.setItemText(6, QCoreApplication.translate("MainPages", u"70-shot", None))
        self.comboBox_op_shot_adv.setItemText(7, QCoreApplication.translate("MainPages", u"80-shot", None))
        self.comboBox_op_shot_adv.setItemText(8, QCoreApplication.translate("MainPages", u"90-shot", None))
        self.comboBox_op_shot_adv.setItemText(9, QCoreApplication.translate("MainPages", u"100-shot", None))

        self.label_8.setText("")
        self.pic_op_matrix1.setText("")
        self.pic_op_matrix2.setText("")
        self.title_label_op_2.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:26pt; font-weight:700;\">\u57fa\u4e8e\u9009\u62e9\u6027\u77e5\u8bc6\u8fc1\u79fb\u7684\u7535\u78c1\u4fe1\u53f7\u8bc6\u522b\u6280\u672f</span></p></body></html>", None))
        self.label_28.setText("")
        self.label_op_data_2.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u9009\u62e9\u6027\u77e5\u8bc6\u8fc1\u79fb\u7b97\u6cd5\u6a21\u578b\u9009\u62e9\uff1a</p></body></html>", None))
        self.comboBox_tl_method.setItemText(0, QCoreApplication.translate("MainPages", u"Baseline", None))
        self.comboBox_tl_method.setItemText(1, QCoreApplication.translate("MainPages", u"Co-Tuning", None))
        self.comboBox_tl_method.setItemText(2, QCoreApplication.translate("MainPages", u"BSS", None))
        self.comboBox_tl_method.setItemText(3, QCoreApplication.translate("MainPages", u"Stochnorm", None))
        self.comboBox_tl_method.setItemText(4, QCoreApplication.translate("MainPages", u"BSS+Stochnorm", None))

        self.label_11.setText("")
        self.label_op_snr_2.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u8c03\u5236\u4fe1\u53f7\u6570\u636e\u96c6\uff1a</p></body></html>", None))
        self.comboBox_tl_dataset.setItemText(0, QCoreApplication.translate("MainPages", u"RML2016.04c", None))

        self.label_12.setText("")
        self.label_op_snr_3.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u4fe1\u566a\u6bd4\uff1a</p></body></html>", None))
        self.comboBox_tl_snr.setItemText(0, QCoreApplication.translate("MainPages", u"6dB-SNR", None))

        self.label_14.setText("")
        self.label_58.setText("")
        self.label_tran4.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u6e90\u57df\u8c03\u5236\u4fe1\u53f7\u7c7b\u522b\uff1a</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">8PSK</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">BPSK</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">CPFSK</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">GFSK</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">PAM4</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">QAM16 </span></p></body></html>", None))
        self.pic_tl_source.setText("")
        self.label_31.setText("")
        self.label_tran5.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u76ee\u6807\u57df\u8c03\u5236\u4fe1\u53f7\u7c7b\u522b\uff1a</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">QPSK</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">AM-DSB</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">AM-SSB</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">QAM64</span><span style=\" font-family:'\u5b8b\u4f53'; font-size:14pt;\">\u3001</span><span style=\" font-family:'Times New Roman,serif'; font-size:14pt;\">WBFM</span></p></body></html>", None))
        self.pic_tl_target.setText("")
        self.label_29.setText("")
        self.label_tran6.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u76ee\u6807\u57df\u8bad\u7ec3\u96c6\u6bcf\u7c7b\u6837\u672c\u6570\u91cf\uff1a</span></p></body></html>", None))
        self.comboBox_tl_shot.setItemText(0, QCoreApplication.translate("MainPages", u"5-shot", None))
        self.comboBox_tl_shot.setItemText(1, QCoreApplication.translate("MainPages", u"10-shot", None))
        self.comboBox_tl_shot.setItemText(2, QCoreApplication.translate("MainPages", u"15-shot", None))
        self.comboBox_tl_shot.setItemText(3, QCoreApplication.translate("MainPages", u"20-shot", None))
        self.comboBox_tl_shot.setItemText(4, QCoreApplication.translate("MainPages", u"25-shot", None))
        self.comboBox_tl_shot.setItemText(5, QCoreApplication.translate("MainPages", u"30-shot", None))
        self.comboBox_tl_shot.setItemText(6, QCoreApplication.translate("MainPages", u"35-shot", None))
        self.comboBox_tl_shot.setItemText(7, QCoreApplication.translate("MainPages", u"40-shot", None))
        self.comboBox_tl_shot.setItemText(8, QCoreApplication.translate("MainPages", u"45-shot", None))
        self.comboBox_tl_shot.setItemText(9, QCoreApplication.translate("MainPages", u"50-shot", None))
        self.comboBox_tl_shot.setItemText(10, QCoreApplication.translate("MainPages", u"55-shot", None))
        self.comboBox_tl_shot.setItemText(11, QCoreApplication.translate("MainPages", u"60-shot", None))
        self.comboBox_tl_shot.setItemText(12, QCoreApplication.translate("MainPages", u"65-shot", None))
        self.comboBox_tl_shot.setItemText(13, QCoreApplication.translate("MainPages", u"70-shot", None))
        self.comboBox_tl_shot.setItemText(14, QCoreApplication.translate("MainPages", u"75-shot", None))
        self.comboBox_tl_shot.setItemText(15, QCoreApplication.translate("MainPages", u"80-shot", None))
        self.comboBox_tl_shot.setItemText(16, QCoreApplication.translate("MainPages", u"85-shot", None))
        self.comboBox_tl_shot.setItemText(17, QCoreApplication.translate("MainPages", u"90-shot", None))
        self.comboBox_tl_shot.setItemText(18, QCoreApplication.translate("MainPages", u"95-shot", None))
        self.comboBox_tl_shot.setItemText(19, QCoreApplication.translate("MainPages", u"100-shot", None))

        self.label_tran6_2.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u6d4b\u8bd5\u96c6\u6837\u672c\u6570\u91cf\u9009\u62e9\uff1a</span></p></body></html>", None))
        self.comboBox_tl_samples.setItemText(0, QCoreApplication.translate("MainPages", u"1-1510", None))
        self.comboBox_tl_samples.setItemText(1, QCoreApplication.translate("MainPages", u"1-500", None))
        self.comboBox_tl_samples.setItemText(2, QCoreApplication.translate("MainPages", u"501-1000", None))
        self.comboBox_tl_samples.setItemText(3, QCoreApplication.translate("MainPages", u"1001-1510", None))
        self.comboBox_tl_samples.setItemText(4, QCoreApplication.translate("MainPages", u"1-200", None))
        self.comboBox_tl_samples.setItemText(5, QCoreApplication.translate("MainPages", u"201-400", None))
        self.comboBox_tl_samples.setItemText(6, QCoreApplication.translate("MainPages", u"401-600", None))
        self.comboBox_tl_samples.setItemText(7, QCoreApplication.translate("MainPages", u"601-800", None))
        self.comboBox_tl_samples.setItemText(8, QCoreApplication.translate("MainPages", u"801-1000", None))
        self.comboBox_tl_samples.setItemText(9, QCoreApplication.translate("MainPages", u"1001-1200", None))
        self.comboBox_tl_samples.setItemText(10, QCoreApplication.translate("MainPages", u"1201-1400", None))
        self.comboBox_tl_samples.setItemText(11, QCoreApplication.translate("MainPages", u"1400-1510", None))

        self.label_20.setText("")
        self.label_37.setText("")
        self.label_57.setText("")
        self.label_30.setText("")
        self.label_tran7.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:18pt; font-weight:700;\">\u7535\u78c1\u4fe1\u53f7\u8bc6\u522b\u51c6\u786e\u7387\uff1a</span></p></body></html>", None))
        self.label_tran8.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u76ee\u6807\u57df\u6837\u672c\u76f4\u63a5\u8bad\u7ec3\uff1a</span></p></body></html>", None))
        self.label_16.setText("")
        self.label_13.setText("")
        self.label_tran8_2.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u7ecf\u9009\u62e9\u6027\u77e5\u8bc6\u8fc1\u79fb\u540e\uff1a</span></p></body></html>", None))
        self.label_17.setText("")
        self.label_15.setText("")
        self.layout_base_matrix.setText("")
        self.label_tran10.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u76ee\u6807\u57df\u6837\u672c\u76f4\u63a5\u8bad\u7ec3\u6df7\u6dc6\u77e9\u9635</span></p></body></html>", None))
        self.layout_adv_matrix.setText("")
        self.label_tran11.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u9009\u62e9\u6027\u77e5\u8bc6\u8fc1\u79fb\u8bad\u7ec3\u6df7\u6dc6\u77e9\u9635</span></p></body></html>", None))
        self.label_61.setText("")
        self.title_label_op_3.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-weight:700;\">\u57fa\u4e8e\u5f3a\u5316\u5b66\u4e60\u548c\u661f\u5ea7\u56fe\u65f6\u9891\u7279\u5f81\u7684\u4fe1\u53f7\u8c03\u5236\u8bc6\u522b</span></p></body></html>", None))
        self.label_32.setText("")
        self.label_60.setText("")
        self.label_36.setText("")
        self.label_op_snr_4.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u8c03\u5236\u4fe1\u53f7\u6570\u636e\u96c6\uff1a</p></body></html>", None))
        self.comboBox_rl_dataset.setItemText(0, QCoreApplication.translate("MainPages", u"RML2016.04c", None))

        self.label_23.setText("")
        self.label_op_snr_5.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u4fe1\u566a\u6bd4\uff1a</p></body></html>", None))
        self.comboBox_rl_snr.setItemText(0, QCoreApplication.translate("MainPages", u"6dB-SNR", None))

        self.label_22.setText("")
        self.label_38.setText("")
        self.label_19.setText("")
        self.radioButton_rl_ori.setText(QCoreApplication.translate("MainPages", u"\u539f\u59cb\u661f\u5ea7\u56fe\u6570\u636e", None))
        self.label_24.setText("")
        self.radioButton_rl_all.setText(QCoreApplication.translate("MainPages", u"\u5168\u90e8\u65f6\u9891\u901a\u9053\u6570\u636e", None))
        self.label_25.setText("")
        self.radioButton_rl_rei.setText(QCoreApplication.translate("MainPages", u"\u5f3a\u5316\u5b66\u4e60\u7b5b\u9009\u6570\u636e", None))
        self.label_21.setText("")
        self.label_35.setText("")
        self.label_26.setText("")
        self.label_op_snr_12.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u6d4b\u8bd5\u96c6\u6837\u672c\uff1a</p></body></html>", None))
        self.comboBox_rl_samples.setItemText(0, QCoreApplication.translate("MainPages", u"1-6483", None))
        self.comboBox_rl_samples.setItemText(1, QCoreApplication.translate("MainPages", u"1-1000", None))
        self.comboBox_rl_samples.setItemText(2, QCoreApplication.translate("MainPages", u"1001-2000", None))
        self.comboBox_rl_samples.setItemText(3, QCoreApplication.translate("MainPages", u"2001-3000", None))
        self.comboBox_rl_samples.setItemText(4, QCoreApplication.translate("MainPages", u"3001-4000", None))
        self.comboBox_rl_samples.setItemText(5, QCoreApplication.translate("MainPages", u"4001-5000", None))
        self.comboBox_rl_samples.setItemText(6, QCoreApplication.translate("MainPages", u"5001-6000", None))
        self.comboBox_rl_samples.setItemText(7, QCoreApplication.translate("MainPages", u"6001-6483", None))

        self.label_34.setText("")
        self.label_18.setText("")
        self.pic_rl_matrix.setText("")
        self.label_tran10_2.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u4fe1\u53f7\u8c03\u5236\u8bc6\u522b\u7ed3\u679c\u7684\u6df7\u6dc6\u77e9\u9635\u56fe\u793a</span></p></body></html>", None))
        self.label_33.setText("")
        self.label_op_snr_6.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u6570\u636e\u9009\u62e9\uff1a</p></body></html>", None))
        self.comboBox_rl_datachoice.setItemText(0, QCoreApplication.translate("MainPages", u"\u968f\u673a", None))
        self.comboBox_rl_datachoice.setItemText(1, QCoreApplication.translate("MainPages", u"GFSK", None))
        self.comboBox_rl_datachoice.setItemText(2, QCoreApplication.translate("MainPages", u"WBFM", None))
        self.comboBox_rl_datachoice.setItemText(3, QCoreApplication.translate("MainPages", u"AM-SSB", None))
        self.comboBox_rl_datachoice.setItemText(4, QCoreApplication.translate("MainPages", u"AM-DSB", None))
        self.comboBox_rl_datachoice.setItemText(5, QCoreApplication.translate("MainPages", u"QPSK", None))
        self.comboBox_rl_datachoice.setItemText(6, QCoreApplication.translate("MainPages", u"QAM16", None))
        self.comboBox_rl_datachoice.setItemText(7, QCoreApplication.translate("MainPages", u"CPFSK", None))
        self.comboBox_rl_datachoice.setItemText(8, QCoreApplication.translate("MainPages", u"BPSK", None))
        self.comboBox_rl_datachoice.setItemText(9, QCoreApplication.translate("MainPages", u"PAM4", None))
        self.comboBox_rl_datachoice.setItemText(10, QCoreApplication.translate("MainPages", u"QAM64", None))
        self.comboBox_rl_datachoice.setItemText(11, QCoreApplication.translate("MainPages", u"8PSK", None))

        self.label_op_snr_8.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u771f\u5b9e\u8c03\u5236\uff1a</p></body></html>", None))
        self.label_rl_real.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><br/></p></body></html>", None))
        self.label_op_snr_7.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u8bc6\u522b\u8c03\u5236\uff1a</p></body></html>", None))
        self.label_rl_rec.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><br/></p></body></html>", None))
        self.label_op_snr_9.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><br/></p></body></html>", None))
        self.pic_rl_data_real.setText("")
        self.pic_rl_data_virtual.setText("")
        self.label_tran10_4.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u4fe1\u53f7\u65f6\u57df\u6ce2\u5f62\u56fe</span></p></body></html>", None))
        self.pic_rl_shipin.setText("")
        self.label_tran10_3.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u4fe1\u53f7\u65f6\u9891\u5206\u89e3\u7279\u5f81\u901a\u9053\u56fe\u793a</span></p></body></html>", None))
        self.label_62.setText("")
        self.title_label_op_4.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-weight:700;\">\u9762\u5411\u7535\u78c1\u4fe1\u53f7\u7f51\u7edc\u7279\u5f81\u7684\u53ef\u89e3\u91ca\u6027\u7814\u7a76</span></p></body></html>", None))
        self.label_39.setText("")
        self.label_41.setText("")
        self.label_op_snr_11.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u8c03\u5236\u4fe1\u53f7\u6570\u636e\u96c6\uff1a</p></body></html>", None))
        self.comboBox_in_dataset.setItemText(0, QCoreApplication.translate("MainPages", u"RML2016.04c", None))

        self.label_42.setText("")
        self.label_op_snr_10.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u4fe1\u566a\u6bd4\uff1a</p></body></html>", None))
        self.comboBox_in_snr.setItemText(0, QCoreApplication.translate("MainPages", u"6dB-SNR", None))

        self.label_43.setText("")
        self.label_op_data_3.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u53ef\u89c6\u5316\u7b97\u6cd5\uff1a</p></body></html>", None))
        self.comboBox_in_method.setItemText(0, QCoreApplication.translate("MainPages", u"Grad-SigCAM++", None))

        self.label_44.setText("")
        self.label_op_data_4.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p>\u53ef\u89c6\u5316\u5c42\uff1a</p></body></html>", None))
        self.comboBox_in_layer.setItemText(0, QCoreApplication.translate("MainPages", u"Conv-1", None))
        self.comboBox_in_layer.setItemText(1, QCoreApplication.translate("MainPages", u"Conv-2", None))
        self.comboBox_in_layer.setItemText(2, QCoreApplication.translate("MainPages", u"Conv-3", None))
        self.comboBox_in_layer.setItemText(3, QCoreApplication.translate("MainPages", u"Conv-4", None))

        self.label_56.setText("")
        self.label_40.setText("")
        self.label_45.setText("")
        self.label_in_mod_1.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">GFSK</span></p></body></html>", None))
        self.pic_in_mod1_1.setText("")
        self.pic_in_mod1_2.setText("")
        self.pic_in_mod1_3.setText("")
        self.label_46.setText("")
        self.label_in_mod_2.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">WBFM</span></p></body></html>", None))
        self.pic_in_mod2_1.setText("")
        self.pic_in_mod2_2.setText("")
        self.pic_in_mod2_3.setText("")
        self.label_47.setText("")
        self.label_in_mod_3.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">AM-SSB</span></p></body></html>", None))
        self.pic_in_mod3_1.setText("")
        self.pic_in_mod3_2.setText("")
        self.pic_in_mod3_3.setText("")
        self.label_48.setText("")
        self.label_in_mod_4.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">AM-DSB</span></p></body></html>", None))
        self.pic_in_mod4_1.setText("")
        self.pic_in_mod4_2.setText("")
        self.pic_in_mod4_3.setText("")
        self.label_49.setText("")
        self.label_in_mod_5.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">QPSK</span></p></body></html>", None))
        self.pic_in_mod5_1.setText("")
        self.pic_in_mod5_2.setText("")
        self.pic_in_mod5_3.setText("")
        self.label_50.setText("")
        self.label_in_mod_6.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">QAM16</span></p></body></html>", None))
        self.pic_in_mod6_1.setText("")
        self.pic_in_mod6_2.setText("")
        self.pic_in_mod6_3.setText("")
        self.label_51.setText("")
        self.label_in_mod_7.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">CPFSK</span></p></body></html>", None))
        self.pic_in_mod7_1.setText("")
        self.pic_in_mod7_2.setText("")
        self.pic_in_mod7_3.setText("")
        self.label_52.setText("")
        self.label_in_mod_8.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">BPSK</span></p></body></html>", None))
        self.pic_in_mod8_1.setText("")
        self.pic_in_mod8_2.setText("")
        self.pic_in_mod8_3.setText("")
        self.label_53.setText("")
        self.label_in_mod_9.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">PAM4</span></p></body></html>", None))
        self.pic_in_mod9_1.setText("")
        self.pic_in_mod9_2.setText("")
        self.pic_in_mod9_3.setText("")
        self.label_54.setText("")
        self.label_in_mod_10.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">QAM64</span></p></body></html>", None))
        self.pic_in_mod10_1.setText("")
        self.pic_in_mod10_2.setText("")
        self.pic_in_mod10_3.setText("")
        self.label_55.setText("")
        self.label_in_mod_11.setText(QCoreApplication.translate("MainPages", u"<html><head/><body><p><span style=\" font-size:14pt;\">8PSK</span></p></body></html>", None))
        self.pic_in_mod11_1.setText("")
        self.pic_in_mod11_2.setText("")
        self.pic_in_mod11_3.setText("")
    # retranslateUi

