from facenet_pytorch import MTCNN, InceptionResnetV1
from src.models.utils import cosine_simularity, sharpen_image


class DocumentFaceChecker:

    def __init__(self):
        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=30,
            image_size=200
        )
        self.embedding = InceptionResnetV1(pretrained='vggface2').eval()

    def check(self, img_RGB):
        boxes, probs = self.mtcnn.detect(img_RGB)

        if probs.shape[0] < 2:
            raise Exception("couldn't find two faces")

        sorted_by_area = sorted(
            boxes,
            key=lambda box: (box[1] - box[3]) * (box[0] - box[2]),
            reverse=True
        )
        faces = self.mtcnn.extract(img_RGB, sorted_by_area, None)
        print(faces[1].shape)
        sharpened_image = sharpen_image(faces[1])
        print(sharpened_image.shape)
        faces[1] = sharpened_image

        print(faces.shape)
        vector_faces = self.embedding(faces)[:2]
        face1, face2 = vector_faces[0].detach(), vector_faces[1].detach()
        return cosine_simularity(face1, face2)

