//1주차-실습1번

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<opencv2/flann/flann.hpp>
#include <iostream>
#include <stdio.h>
#define RED Scalar(0,0,255)

using namespace cv;
Mat boxFindImg(std::vector<DMatch> good_match, std::vector<KeyPoint> img1keypoint, std::vector<KeyPoint> camkeypoint, Mat img1, Mat finalOutputImg);

int main()
{
	//검출 -> 기술 -> 매칭
	//*****************각자 검출하고싶은 물체의 이미지 파일 위치를 대입
	Mat cam, img1;
	img1 = imread("이미지 주소를 넣으세요", IMREAD_GRAYSCALE);
	//카메라 준비
	VideoCapture cap;
	cap.open(0); //0번째 카메라 open

	if (!cap.isOpened())
	{
		std::cout << "카메라 작동 불가" << std::endl;
		return -1;
	}

	if (!(img1.data))
	{
		printf("이미지를 로드할 수 없습니다.");
		return 0;
	}
	//결과나 진행상황을 보여줄 창 미리 생성
	namedWindow("camera", WINDOW_AUTOSIZE);
	namedWindow("img1의 키포인트");
	namedWindow("img2의 키포인트");
	namedWindow("매칭 결과");

	SIFT instance_FeatureDetector;//검출을 위한 인스턴스 생성
	std::vector<KeyPoint> img1keypoint, camkeypoint;
	instance_FeatureDetector.detect(img1, img1keypoint);//img1에 특징점을 img1keypoint에 저장

	for (;;) {
		cap.read(cam); 
		imshow("camera", cam);

		//esc키를 누르면 반복 종료
		int c = -1;
		c = waitKey(33);
		if (c == 27)//esc키를 누르면 반복 종료
			break;
		else if (c < 0)//esc를 제외한 아무키나 누르면 됨
			continue;

		//*****************카메라 이미지 준비 완료 1.검출 with sift (SiftFeatureDetector)
		instance_FeatureDetector.detect(cam, camkeypoint); //camera에서 읽어온 이미지에서의 특징점을 camkeypoint에 저장

			//이미지에서 키포인트를 검출했으므로 키포인트를 시각적으로 보여줌
		Mat displayOfImg1;
		drawKeypoints(img1, img1keypoint, displayOfImg1, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//빨간색 DRAW_RICH_KEYPOINTS로 나타냄
		imshow("img1의 키포인트", displayOfImg1);

		Mat displayOfImg2;
		drawKeypoints(cam, camkeypoint, displayOfImg2, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//빨간색 DRAW_RICH_KEYPOINTS로 나타냄
		imshow("img2의 키포인트", displayOfImg2);

		//*****************1.기술 with sift (SiftDescriptorExtractor)
		SIFT instance_Descriptor;
		Mat img1outputarray, camoutputarray;
		instance_Descriptor.compute(img1, img1keypoint, img1outputarray);
		instance_Descriptor.compute(cam, camkeypoint, camoutputarray);

		std::cout << "기술 완료" << std::endl;

		//*****************1.매칭 with sift (SiftDescriptorExtractor)
		FlannBasedMatcher FLANNmatcher;
		std::vector<DMatch> match;
		FLANNmatcher.match(img1outputarray, camoutputarray, match);

		if (!(match.size()))
		{
			std::cout << "키포인트 매칭 불가!" << std::endl;
			return -1;
		}

		//매칭된 쌍들 중에서 유클리드 거리를 기준으로 굿매치(믿을만한 매칭)를 추출
		double maxd = 0; double mind = match[0].distance;
		for (int i = 0; i < match.size(); i++)
		{
			double dist = match[i].distance;
			if (dist < mind) mind = dist;
			if (dist > maxd) maxd = dist;
		}
		std::vector<DMatch> good_match;
		for (int i = 0; i < match.size(); i++)
			if (match[i].distance <= max(2 * mind, 0.02)) good_match.push_back(match[i]);

		Mat finalOutputImg;
		std::cout << "match 의 갯수는: " << match.size() << std::endl;
		std::cout << "good match 의 갯수는: " << good_match.size() << std::endl;

		drawMatches(img1, img1keypoint, cam, camkeypoint, good_match, finalOutputImg, Scalar(150, 30, 200), Scalar(0, 0, 255), std::vector< char >(), DrawMatchesFlags::DEFAULT);
		finalOutputImg = boxFindImg(good_match, img1keypoint, camkeypoint, img1, finalOutputImg);
		imshow("매칭 결과", finalOutputImg);
	}

	return 0;
}
Mat boxFindImg(std::vector<DMatch> good_match, std::vector<KeyPoint> img1keypoint, std::vector<KeyPoint> camkeypoint, Mat img1, Mat finalOutputImg)
{
	std::vector<Point2f> model_pt;
	std::vector<Point2f> scene_pt;
	for (int i = 0; i < good_match.size(); i++) {
		model_pt.push_back(img1keypoint[good_match[i].queryIdx].pt);
		scene_pt.push_back(camkeypoint[good_match[i].trainIdx].pt);
	}
	Mat H = findHomography(model_pt, scene_pt, CV_RANSAC);

	std::vector<Point2f> model_corner(4);
	model_corner[0] = cvPoint(0, 0);
	model_corner[1] = cvPoint(img1.cols, 0);
	model_corner[2] = cvPoint(img1.cols, img1.rows);
	model_corner[3] = cvPoint(0, img1.rows);

	std::vector<Point2f> scene_corner(4);
	perspectiveTransform(model_corner, scene_corner, H);
	//좌표에 투영변환 H를 적용하여 장면 영상에 나타날 좌표 scene_corner를 계산한다

	Point2f p(img1.cols, 0); //4개의 선분영상 그려넣음으로써 인식 결과 표시
							 //하나의 윈도우에 모델 영상과 장면 영상을 같이 그려넣었기때문에 모델 영상의 너비만큼 이동시켜 그리기 위해 
							 //Point2f 형의 p를 더해주었다.
	line(finalOutputImg, scene_corner[0] + p, scene_corner[1] + p, RED, 3);
	line(finalOutputImg, scene_corner[1] + p, scene_corner[2] + p, RED, 3);
	line(finalOutputImg, scene_corner[2] + p, scene_corner[3] + p, RED, 3);
	line(finalOutputImg, scene_corner[3] + p, scene_corner[0] + p, RED, 3);

	return finalOutputImg;
}
