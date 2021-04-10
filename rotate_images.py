import cv2, os

def flip_images():
	gest_folder = ""
	images_labels = []
	images = []
	labels = []
	for g_id in os.listdir(gest_folder):
		for i in range(1200):
			path = ""
			new_path = ""
			print(path)
			img = cv2.imread(path, 0)
			img = cv2.flip(img, 1)
			cv2.imwrite(new_path, img)

flip_images()
