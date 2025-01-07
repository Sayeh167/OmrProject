import cv2
import numpy as np
import utils
import time

pathImage = "c.jpg"
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]

while True:
    start = time.time()
    img = cv2.imread(pathImage)

    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)  # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY

    try:
        # FIND ALL COUNTOURS
        imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # FIND ALL CONTOURS
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS
        rectCon = utils.rectContour(contours)  # FILTER FOR RECTANGLE CONTOURS
        biggestPoints = utils.getCornerPoints(rectCon[0])  # GET CORNER POINTS OF THE BIGGEST RECTANGLE
        gradePoints = utils.getCornerPoints(rectCon[1])  # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE

        if biggestPoints.size != 0 and gradePoints.size != 0:
            # BIGGEST RECTANGLE WARPING
            biggestPoints = utils.reorder(biggestPoints)
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)
            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))


            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)
            gradePoints = utils.reorder(gradePoints)
            ptsG1 = np.float32(gradePoints)
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))


            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = utils.splitBoxes(imgThresh)

            countR = 0
            countC = 0
            myPixelVal = np.zeros((questions, choices))
            for image in boxes:
                # cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if (countC == choices):
                    countC = 0
                    countR += 1

            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])

            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)

            score = (sum(grading) / questions) * 100  # FINAL GRADE
            print("SCORE",score)


            utils.showAnswers(imgWarpColored, myIndex, grading, ans)
            utils.drawGrid(imgWarpColored)
            imgRawDrawings = np.zeros_like(imgWarpColored)
            utils.showAnswers(imgRawDrawings, myIndex, grading, ans)
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

            imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)
            cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100)
                        , cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            # SHOW ANSWERS AND GRADE ON FINAL IMAGE
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

            imageArray = ([img, imgGray, imgCanny, imgContours],
                          [imgBigContour, imgThresh, imgWarpColored, imgFinal])
            cv2.imshow("Final Result", imgFinal)

    except:
        print("Image cannot be processed.")
        imageArray = ([img, imgGray, imgCanny, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    end = time.time()
    print("Time Taken: ", end-start)
    # LABELS FOR DISPLAY
    lables = [["Original", "Gray", "Edges", "Contours"],
              ["Biggest Contour", "Threshold", "Warpped", "Final"]]

    stackedImage = utils.stackImages(imageArray, 0.5, lables)

    cv2.imshow('Result', stackedImage)
    cv2.waitKey(5000)