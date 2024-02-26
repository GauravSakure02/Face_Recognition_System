import cv2
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = 'C:\\Users\\Gaurav\\OneDrive\\Desktop\\C++\\Face-recognition-using-KNN-master\\data\\'
file_name = input('Enter the name :')

while True:
	ret, frame = cap.read()

	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if (len(faces)) == 0:
		continue

		

	faces = sorted(faces, key=lambda f: f[2]*f[3])

	face_section = np.ones((100,100)) * 255
	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the required face) : Region of Interest
		offset = 10

		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))
	
	cv2.imshow("Frame", frame)
	cv2.imshow("Face Section", face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# convert our face list array into numpy array

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))



#save this data into file system
np.save(dataset_path+file_name+ '.npy',face_data)
print("data sucessfully saved!!")
cap.release()
cv2.destroyAllWindows()
