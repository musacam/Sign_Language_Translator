import cv2, os

def flip_images():
	gest_folder = "data/train"
	for g_id in os.listdir(gest_folder):
		for i in range(900):
			path = gest_folder+"/"+g_id+"/"+str(i)+".jpg"
			new_path = gest_folder+"/"+g_id+"/"+str(i+900)+".jpg"
			print(path)
			img = cv2.imread(path, 0)
			img = cv2.flip(img, 1)
			cv2.imwrite(new_path, img)

flip_images()