import numpy as np
import struct
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 训练集文件路径
train_images_idx3_ubyte_file = './datafile/train-images-idx3-ubyte/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './datafile/train-labels-idx1-ubyte/train-labels.idx1-ubyte'

# 测试集文件路径
test_images_idx3_ubyte_file = './datafile/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './datafile/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    """解析idx3文件的通用函数"""
    with open(idx3_ubyte_file, 'rb') as f:
        bin_data = f.read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, image_size))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """解析idx1文件的通用函数"""
    with open(idx1_ubyte_file, 'rb') as f:
        bin_data = f.read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def main():
    np.random.seed()  # 重新设置随机种子

    # 加载训练数据和测试数据
    print("Loading data...")
    X_train = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    y_train = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    X_test = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    y_test = decode_idx1_ubyte(test_labels_idx1_ubyte_file)

    # 数据归一化
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 随机打乱训练数据
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    # 创建随机森林模型
    print("Training model...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # 进行预测
    print("Evaluating model...")
    y_pred = rf_classifier.predict(X_test)

    # 计算正确率
    correct_predictions = sum(y_pred == y_test)
    total_predictions = len(y_test)
    overall_accuracy = correct_predictions / total_predictions
    print("Overall Accuracy: {:.2f}%".format(overall_accuracy * 100))
    print("正确率：{}/{}".format(correct_predictions, total_predictions))

    # 打印性能报告
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 可视化部分测试结果
    print("Visualizing predictions...")
    visualize_predictions(X_test, y_test, y_pred, rf_classifier)


def visualize_predictions(X_test, y_test, y_pred, rf_classifier, num_images=8):
    # 随机选择 num_images 个索引
    random_indices = np.random.choice(len(X_test), num_images, replace=False)
    fig, axes = plt.subplots(2, num_images // 2, figsize=(20, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        index = random_indices[i]
        image = X_test[index].reshape(28, 28)
        true_label = y_test[index]
        predicted_label = y_pred[index]

        # 获取预测的概率分布
        proba = rf_classifier.predict_proba([X_test[index]])[0]
        confidence = np.max(proba)

        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.2f}")
        ax.axis('off')

        # 如果预测错误，用红色标题标记
        if true_label != predicted_label:
            ax.set_title(f"True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.2f}", color='red')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
