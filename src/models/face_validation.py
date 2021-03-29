from facenet_pytorch import MTCNN, InceptionResnetV1
from src.models.utils import cosine_simularity


class DocumentFaceChecker:

    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True, min_face_size=10, image_size=200)
        self.embedding = InceptionResnetV1(pretrained='vggface2').eval()

    def check(self, img_RGB):
        faces = self.mtcnn(img_RGB)

        if faces.shape[0] < 2:
            raise Exception("couldn't find two faces")

        print(faces.shape)
        vector_faces = self.embedding(faces)[:2]
        face1, face2 = vector_faces[0].detach(), vector_faces[1].detach()
        return cosine_simularity(face1, face2)

