import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from nueralnet import NueraLNet, prepare_image

class ImagePredictor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("图像识别结果")
        self.model = NueraLNet([28 * 28, 30, 15, 10])
        self.model.read_training()
        self.img_tk = None  # 保持对 PhotoImage 对象的引用

        self.canvas = tk.Canvas(self, width=280, height=280)
        self.canvas.pack()

        self.result_label = tk.Label(self, font=("Arial", 16))
        self.result_label.pack()

        self.choose_button = tk.Button(self, text="选择图片", command=self.choose_and_predict)
        self.choose_button.pack()

    def choose_and_predict(self):
        # 弹出文件选择对话框
        file_path = filedialog.askopenfilename(title='选择图片文件',
                                               filetypes=[('PNG files', '*.png'), ('All files', '*.*')])
        if not file_path:
            print("未选择任何文件")
            return

        # 处理选择的图像
        image_array = prepare_image(file_path)

        # 使用模型进行预测
        recognized_digit = self.model.predict(image_array)

        # 打开图像并转换为tkinter兼容的格式
        img = Image.open(file_path)
        img = img.resize((280, 280), Image.LANCZOS)  # 调整图像大小
        self.img_tk = ImageTk.PhotoImage(img)

        # 在 Canvas 上显示图像
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        # 显示识别结果
        self.result_label.config(text=f"识别结果为: {recognized_digit}")

if __name__ == "__main__":
    app = ImagePredictor()
    app.mainloop()
