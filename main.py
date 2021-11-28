# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

# IMPORT PACKAGES AND MODULES
# ///////////////////////////////////////////////////////////////
from gui.uis.windows.main_window.functions_main_window import *
import sys
import os

# IMPORT QT CORE
# ///////////////////////////////////////////////////////////////
from qt_core import *

# IMPORT SETTINGS
# ///////////////////////////////////////////////////////////////
from gui.core.json_settings import Settings

# IMPORT PY ONE DARK WINDOWS
# ///////////////////////////////////////////////////////////////
# MAIN WINDOW
from gui.uis.windows.main_window import *

# IMPORT PY ONE DARK WIDGETS
# ///////////////////////////////////////////////////////////////
from gui.widgets import *

# ADJUST QT FONT DPI FOR HIGHT SCALE AN 4K MONITOR
# ///////////////////////////////////////////////////////////////
os.environ["QT_FONT_DPI"] = "96"
# IF IS 4K MONITOR ENABLE 'os.environ["QT_SCALE_FACTOR"] = "2"'

# IMPORT MAIN INFERENCE PROGRAM
from app.inference_UI import nextRLdata, showDataset, inference_OP_TL, inference_RL, nextRLdata, showCAM
import cv2
# MAIN WINDOW
# ///////////////////////////////////////////////////////////////
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # SETUP MAIN WINDOw
        # Load widgets from "gui\uis\main_window\ui_main.py"
        # ///////////////////////////////////////////////////////////////
        self.ui = UI_MainWindow()
        self.ui.setup_ui(self)

        # LOAD SETTINGS
        # ///////////////////////////////////////////////////////////////
        settings = Settings()
        self.settings = settings.items

        # SETUP MAIN WINDOW
        # ///////////////////////////////////////////////////////////////
        self.hide_grips = True # Show/Hide resize grips
        SetupMainWindow.setup_gui(self)

        # SHOW MAIN WINDOW
        # ///////////////////////////////////////////////////////////////
        self.show()

    # LEFT MENU BTN IS CLICKED
    # Run function when btn is clicked
    # Check funtion by object name / btn_id
    # ///////////////////////////////////////////////////////////////
    def btn_clicked(self):

        # GET BT CLICKED
        btn = SetupMainWindow.setup_btns(self)

        # Remove Selection If Clicked By "btn_close_left_column"
        if btn.objectName() != "btn_settings":
            self.ui.left_menu.deselect_all_tab()

        # Get Title Bar Btn And Reset Active         
        top_settings = MainFunctions.get_title_bar_btn(self, "btn_top_settings")
        top_settings.set_active(False)

        # LEFT MENU
        # ///////////////////////////////////////////////////////////////
        
        # 主页 按钮转跳
        if btn.objectName() == "btn_home":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            # Load Page 1
            MainFunctions.set_page(self, self.ui.load_pages.page_home)

        # 迁移学习 按钮转跳
        if btn.objectName() == "btn_TL":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())
                    
            # 预加载源域，目标域图像
            # //////////////////////////////////////////////////////////////////////////
            size = max(self.ui.load_pages.pic_tl_source.width(), self.ui.load_pages.pic_tl_source.height())
            img_source = cv2QPix(cv2.imread("./gui/images/apps_images/transfer_source.png"), 0).scaled(size, int(size/6))
            img_target = cv2QPix(cv2.imread("./gui/images/apps_images/transfer_target.png"), 0).scaled(size, int(size/6))
            self.ui.load_pages.pic_tl_source.setPixmap(img_source)
            self.ui.load_pages.pic_tl_target.setPixmap(img_target)

            # Load Page 2
            MainFunctions.set_page(self, self.ui.load_pages.page_TL)

        # 模型优化 按钮转跳
        if btn.objectName() == "btn_OP":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            # Load Page 3 
            MainFunctions.set_page(self, self.ui.load_pages.page_OP)

        # 强化学习 按钮转跳
        if btn.objectName() == "btn_RL":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            # Load Page 3 
            MainFunctions.set_page(self, self.ui.load_pages.page_RL)

        # LOAD USER PAGE
        if btn.objectName() == "btn_IN":
            # Select Menu
            self.ui.left_menu.select_only_one(btn.objectName())

            # Load Page 3 
            MainFunctions.set_page(self, self.ui.load_pages.page_IN)

        # BOTTOM INFORMATION
        if btn.objectName() == "btn_info":
            # CHECK IF LEFT COLUMN IS VISIBLE
            if not MainFunctions.left_column_is_visible(self):
                self.ui.left_menu.select_only_one_tab(btn.objectName())

                # Show / Hide
                MainFunctions.toggle_left_column(self)
                self.ui.left_menu.select_only_one_tab(btn.objectName())
            else:
                if btn.objectName() == "btn_close_left_column":
                    self.ui.left_menu.deselect_all_tab()
                    # Show / Hide
                    MainFunctions.toggle_left_column(self)
                
                self.ui.left_menu.select_only_one_tab(btn.objectName())

            # Change Left Column Menu
            if btn.objectName() != "btn_close_left_column":
                MainFunctions.set_left_column_menu(
                    self, 
                    menu = self.ui.left_column.menus.menu_2,
                    title = "Info tab",
                    icon_path = Functions.set_svg_icon("icon_info.svg")
                )

        # SETTINGS LEFT
        if btn.objectName() == "btn_settings" or btn.objectName() == "btn_close_left_column":
            # CHECK IF LEFT COLUMN IS VISIBLE
            if not MainFunctions.left_column_is_visible(self):
                # Show / Hide
                MainFunctions.toggle_left_column(self)
                self.ui.left_menu.select_only_one_tab(btn.objectName())
            else:
                if btn.objectName() == "btn_close_left_column":
                    self.ui.left_menu.deselect_all_tab()
                    # Show / Hide
                    MainFunctions.toggle_left_column(self)
                self.ui.left_menu.select_only_one_tab(btn.objectName())

            # Change Left Column Menu
            if btn.objectName() != "btn_close_left_column":
                MainFunctions.set_left_column_menu(
                    self, 
                    menu = self.ui.left_column.menus.menu_1,
                    title = "Settings Left Column",
                    icon_path = Functions.set_svg_icon("icon_settings.svg")
                )
        
        # TITLE BAR MENU
        # ///////////////////////////////////////////////////////////////
        
        # SETTINGS TITLE BAR
        if btn.objectName() == "btn_top_settings":
            # Toogle Active
            if not MainFunctions.right_column_is_visible(self):
                btn.set_active(True)

                # Show / Hide
                MainFunctions.toggle_right_column(self)
            else:
                btn.set_active(False)

                # Show / Hide
                MainFunctions.toggle_right_column(self)

            # Get Left Menu Btn            
            top_settings = MainFunctions.get_left_menu_btn(self, "btn_settings")
            top_settings.set_active_tab(False)            

        # DEBUG
        print(f"Button {btn.objectName()}, clicked!")

    # LEFT MENU BTN IS RELEASED
    # Run function when btn is released
    # Check funtion by object name / btn_id
    # ///////////////////////////////////////////////////////////////
    def btn_released(self):
        # GET BT CLICKED
        btn = SetupMainWindow.setup_btns(self)

        # DEBUG
        print(f"Button {btn.objectName()}, released!")

    # RESIZE EVENT
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        SetupMainWindow.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()
        
    # 后期新增的按钮逻辑
    def pages_btn_clicked(self, menu, btn_name):
        ''' 
            自定义按钮点击事件 
        '''
        # 混淆矩阵是否归一化 
        mat_norm = self.ui.toggle_mat_norm.isChecked()      

        '''1. 优化界面按钮事件 '''
        if menu == "optimize":
            # 优化下拉菜单状态获取
            op_dataset = self.ui.load_pages.comboBox_op_data.currentText()
            op_snr = self.ui.load_pages.comboBox_op_snr.currentText()
            op_test_samples = self.ui.load_pages.comboBox_op_sample.currentText().split('-')
            op_shot_base = self.ui.load_pages.comboBox_op_shot_base.currentText().split('-')[0]
            op_shot_adv = self.ui.load_pages.comboBox_op_shot_adv.currentText().split('-')[0]
            op_adv_method = self.ui.load_pages.comboBox_op_method.currentText().split('-')[0]
            
            print(f"op_dataset: {op_dataset}, op_snr: {op_snr}")
            print(f"op_shot_base: {op_shot_base}, op_test_samples: {op_test_samples}")
            print(f"op_shot_adv: {op_shot_adv}, op_adv_method: {op_adv_method}")
            print(f"Normalized confusion matrix: {mat_norm}")
            
            # //////////////////////////////////////////////////////////////////////////////
            # 数据可视化按钮事件
            if btn_name == "pushButton_op_data":

                imgs = showDataset(
                    menu = "optimize",
                    idx_range = op_test_samples,
                    img_num = 8
                )
                for idx, img in enumerate(imgs):
                    # width = eval(f"self.ui.load_pages.pic_op_data_{idx+1}.width()")
                    # height = eval(f"self.ui.load_pages.pic_op_data_{idx+1}.height()")
                    pic_pos = eval(f"self.ui.load_pages.pic_op_data_{idx+1}.setPixmap")
                    img = cv2QPix(img,0).scaled(300,150)
                    pic_pos(img)
                    cv2.waitKey(100)
            # //////////////////////////////////////////////////////////////////////////////
            # Baseline测试按钮事件
            elif btn_name == "pushButton_op_base_test":
                self.ui.circular_progress_op_acc_base.set_value(0)
                # QApplication.processEvents()  # 刷新
                acc, confusion_matrix = inference_OP_TL(
                    menu = "optimize",
                    method = "Baseline",
                    dataset = "RML2016.04c", 
                    idx_range = op_test_samples,
                    shot_num = op_shot_base,
                    infer_mode = 0,
                    mat_norm = mat_norm
                )
                self.ui.circular_progress_op_acc_base.set_value(float(str(round(acc, 2))))
                size = max(self.ui.load_pages.pic_op_matrix1.width(), self.ui.load_pages.pic_op_matrix1.height())
                confusion_matrix = cv2QPix(confusion_matrix).scaled(size, size)
                self.ui.load_pages.pic_op_matrix1.setPixmap(confusion_matrix)
            # //////////////////////////////////////////////////////////////////////////////
            # 优化后网络测试按钮事件
            elif btn_name == "pushButton_op_adv_test":
                self.ui.circular_progress_op_acc_adv.set_value(0)
                acc, confusion_matrix = inference_OP_TL(
                    menu = "optimize",
                    method = op_adv_method,
                    dataset = "RML2016.04c", 
                    idx_range = op_test_samples,
                    shot_num = op_shot_adv,
                    infer_mode = 0,
                    mat_norm = mat_norm
                )
                self.ui.circular_progress_op_acc_adv.set_value(float(str(round(acc, 2))))
                size = max(self.ui.load_pages.pic_op_matrix2.width(), self.ui.load_pages.pic_op_matrix2.height())
                confusion_matrix = cv2QPix(confusion_matrix).scaled(size, size)
                self.ui.load_pages.pic_op_matrix2.setPixmap(confusion_matrix)
            else:
                pass

        '''2. 迁移界面按钮事件 '''
        if menu == "transfer":
            # 迁移下拉菜单状态获取
            tl_dataset = self.ui.load_pages.comboBox_tl_dataset.currentText()
            tl_snr = self.ui.load_pages.comboBox_tl_snr.currentText()
            tl_test_samples = self.ui.load_pages.comboBox_tl_samples.currentText().split('-')
            tl_shot = self.ui.load_pages.comboBox_tl_shot.currentText().split('-')[0]
            tl_adv_method = self.ui.load_pages.comboBox_tl_method.currentText()
            
            print(f"tl_dataset: {tl_dataset}, tl_snr: {tl_snr}")
            print(f"tl_shot: {tl_shot}, tl_test_samples: {tl_test_samples}")
            print(f"tl_adv_method: {tl_adv_method}")
            print(f"Normalized confusion matrix: {mat_norm}")

            # //////////////////////////////////////////////////////////////////////////////
            # 直接训练后测试按钮事件
            if btn_name == "pushButton_tl_base_test":
                self.ui.circular_progress_tl_acc_base.set_value(0)
                acc, confusion_matrix = inference_OP_TL(
                    menu = "transfer",
                    # method = "no-transfer",
                    method = "Baseline",
                    dataset = "RML2016.04c", 
                    idx_range = tl_test_samples,
                    shot_num = tl_shot,
                    infer_mode = 1,
                    mat_norm = mat_norm
                )
                self.ui.circular_progress_tl_acc_base.set_value(float(str(round(acc, 2))))
                size = max(self.ui.load_pages.layout_base_matrix.width(), self.ui.load_pages.layout_base_matrix.height())
                confusion_matrix = cv2QPix(confusion_matrix).scaled(size, size)
                self.ui.load_pages.layout_base_matrix.setPixmap(confusion_matrix)
            
            # //////////////////////////////////////////////////////////////////////////////
            # 迁移训练后测试按钮事件
            if btn_name == "pushButton_tl_adv_test":
                self.ui.circular_progress_tl_acc_adv.set_value(0)
                acc, confusion_matrix = inference_OP_TL(
                    menu = "transfer",
                    # method = "tl_adv_method",
                    method = "Baseline",
                    dataset = "RML2016.04c", 
                    idx_range = tl_test_samples,
                    shot_num = tl_shot,
                    infer_mode = 1,
                    mat_norm = mat_norm
                )
                self.ui.circular_progress_tl_acc_adv.set_value(float(str(round(acc, 2))))
                size = max(self.ui.load_pages.layout_adv_matrix.width(), self.ui.load_pages.layout_adv_matrix.height())
                confusion_matrix = cv2QPix(confusion_matrix).scaled(size, size)
                self.ui.load_pages.layout_adv_matrix.setPixmap(confusion_matrix)
        
        '''3. 强化学习界面按钮事件 '''
        if menu == "reinforce":
            # 强化下拉菜单状态获取
            rl_dataset = self.ui.load_pages.comboBox_rl_dataset.currentText()
            rl_snr = self.ui.load_pages.comboBox_rl_snr.currentText()
            rl_test_samples = self.ui.load_pages.comboBox_rl_samples.currentText().split('-')
            rl_data_choice = self.ui.load_pages.comboBox_rl_datachoice.currentText()
            
            if self.ui.load_pages.radioButton_rl_ori.isChecked():
                rl_method = "stars"
            elif self.ui.load_pages.radioButton_rl_all.isChecked():
                rl_method = "stars_haar"
            elif self.ui.load_pages.radioButton_rl_rei.isChecked():
                rl_method = "stars_haar_reinforcement"
            else:
                rl_method = "None"

            print(f"rl_dataset: {rl_dataset}, rl_snr: {rl_snr}")
            print(f"rl_test_samples: {rl_test_samples}")
            print(f"rl_adv_method: {rl_method}")
            print(f"rl_data_choice: {rl_data_choice}")
            print(f"Normalized confusion matrix: {mat_norm}")

            # //////////////////////////////////////////////////////////////////////////////
            # 测试按钮事件
            if btn_name == "pushButton_rl_test":
                self.ui.circular_progress_rl_acc.set_value(0)
                acc, confusion_matrix = inference_RL(
                    method = rl_method,
                    idx_range = rl_test_samples,
                    mat_norm = mat_norm
                )
                self.ui.circular_progress_rl_acc.set_value(float(str(round(acc, 2))))
                size = max(self.ui.load_pages.pic_rl_matrix.width(), self.ui.load_pages.pic_rl_matrix.height())
                confusion_matrix = cv2QPix(confusion_matrix).scaled(size, size)
                self.ui.load_pages.pic_rl_matrix.setPixmap(confusion_matrix)
            
            # //////////////////////////////////////////////////////////////////////////////
            # 下一个样本按钮事件
            if btn_name == "pushButton_rl_next":
                (label1, label2),(starImg, realImg, virtualImg) = nextRLdata(rl_method, rl_data_choice)

                self.ui.load_pages.label_rl_real.setText(label1)
                self.ui.load_pages.label_rl_rec.setText(label2)

                # size = max(self.ui.load_pages.pic_rl_shipin.width(), self.ui.load_pages.pic_rl_shipin.height()) * 0.9
                starImg = cv2QPix(starImg, 0).scaled(400, 400)
                self.ui.load_pages.pic_rl_shipin.setPixmap(starImg)

                # size = max(self.ui.load_pages.pic_rl_data_real.width(), self.ui.load_pages.pic_rl_data_real.height()) * 0.9
                realImg = cv2QPix(realImg, 0).scaled(500, 150)
                self.ui.load_pages.pic_rl_data_real.setPixmap(realImg)
                virtualImg = cv2QPix(virtualImg, 0).scaled(500, 150)
                self.ui.load_pages.pic_rl_data_virtual.setPixmap(virtualImg)

        '''4. 网络可解释按钮事件 '''
        if menu == "interpretability":
            # 强化下拉菜单状态获取
            in_dataset = self.ui.load_pages.comboBox_in_dataset.currentText()
            in_snr = self.ui.load_pages.comboBox_in_snr.currentText()
            in_method = self.ui.load_pages.comboBox_in_method.currentText()
            in_vis_layer = self.ui.load_pages.comboBox_in_layer.currentText().split('-')[1]
            
            print(f"in_dataset: {in_dataset}, in_snr: {in_snr}")
            print(f"in_method: {in_method}")
            print(f"in_vis_layer: {in_vis_layer}")

            # //////////////////////////////////////////////////////////////////////////////
            # 直接训练后测试按钮事件
            if btn_name == "pushButton_in_next":
                images = showCAM(
                    dataset = in_dataset,
                    layer = int(in_vis_layer),
                    imgLen = 3
                )
                for row in range(len(images)):
                    for col in range(len(images[row])):
                        pic_pos = eval(f"self.ui.load_pages.pic_in_mod{row+1}_{col+1}.setPixmap")
                        img = cv2QPix(images[row][col],0).scaled(500,250)
                        pic_pos(img)
                        cv2.waitKey(100)
        # DEBUG
        print(f"Button {btn_name}, clicked!")

# SETTINGS WHEN TO START
# Set the initial class and also additional parameters of the "QApplication" class
# ///////////////////////////////////////////////////////////////
if __name__ == "__main__":
    # APPLICATION
    # ///////////////////////////////////////////////////////////////
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()

    # EXEC APP
    # ///////////////////////////////////////////////////////////////
    sys.exit(app.exec_())