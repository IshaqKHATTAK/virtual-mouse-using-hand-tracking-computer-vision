#  1. hand detector
#  2. find position
#  3. find hands
#  4. find distance


import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHand=2, modelComplex=1, minConfi=0.5,
                 trackConfi=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.modelComplex = modelComplex
        self.minConfi = minConfi
        self.trackConfi = trackConfi
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHand, self.modelComplex, self.minConfi, self.trackConfi)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.res = self.hands.process(img_rgb)
        # res.multi_hand_landmarks contain the coordinate of hands detected
        if self.res.multi_hand_landmarks:
            for handlm in self.res.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handlm,
                                               self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, img, handNo=0):
        lmlist = []
        if self.res.multi_hand_landmarks:
            handlm = self.res.multi_hand_landmarks[handNo]

            for id, lm in enumerate(handlm.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), (lm.y * h)
                lmlist.append([id, cx, cy])
        return lmlist
        # we multiply the x and y coordinate of the landmark with the h and w
        # becoz we want to find the pixel location instead of decimal


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 400)
    cap.set(4, 600)
    detector = handDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        lmhands = detector.findPosition(frame)
        if len(lmhands) != 0:
            print(lmhands[4])
        cv2.imshow('gray', frame)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
