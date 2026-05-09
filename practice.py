import cv2
import os

class CowNosePreprocessor:
    def __init__(self, input_folder, output_folder, size=200):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.size = size

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def resize_with_padding(self, image):
        h, w = image.shape

        scale = min(self.size / w, self.size / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        top = (self.size - new_h) // 2
        bottom = self.size - new_h - top
        left = (self.size - new_w) // 2
        right = self.size - new_w - left

        final_image = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=0
        )

        return final_image

    def preprocess_one(self, filename):
        input_path = os.path.join(self.input_folder, filename)

        image = cv2.imread(input_path)

        if image is None:
            print(filename, "불러오기 실패")
            return

        # 마우스로 비문 영역 선택
        roi = cv2.selectROI("Select Cow Nose Area", image)

        cv2.destroyAllWindows()

        x, y, w, h = roi

        if w == 0 or h == 0:
            print(filename, "영역 선택 안 됨")
            return

        cropped = image[y:y+h, x:x+w]

        # 흑백 변환
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # 노이즈 제거
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 비율 유지 resize
        final_image = self.resize_with_padding(gray)

        name, ext = os.path.splitext(filename)

        output_path = os.path.join(
            self.output_folder,
            name + ".png"
        )

        cv2.imwrite(output_path, final_image)

        print(filename, "전처리 완료")

    def preprocess_all(self):
        for filename in os.listdir(self.input_folder):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                self.preprocess_one(filename)


preprocessor = CowNosePreprocessor(
    input_folder="input",
    output_folder="output",
    size=200
)

preprocessor.preprocess_all()