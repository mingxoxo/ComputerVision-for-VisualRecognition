#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<opencv2/flann/flann.hpp>
#include<opencv2/imgproc/imgproc.hpp> //이미지 사이즈 조정을 위해 추가로 include
#include <iostream>
#include <stdio.h>

using namespace cv;

int main()
{
	//검출 -> 기술 -> 매칭
	//*****************각자 검출하고싶은 물체의 이미지 파일 위치를 대입
	Mat img1, img2;
	//연산속도를 높이기 위해 IMREAD_GRAYSCALE로 gray이미지로 읽어오고 사이즈를 줄인다.
	img1 = imread("이미지 주소를 넣으세요", IMREAD_GRAYSCALE);
	img2 = imread("이미지 주소를 넣으세요", IMREAD_GRAYSCALE);
	Size size(img1.cols / 2, img1.rows / 2);
	resize(img1, img1, size);
	resize(img2, img2, size);


	if (!(img1.data && img2.data))
	{
		printf("이미지를 로드할 수 없습니다.");
		return 0;
	}
	//결과나 진행상황을 보여줄 창 미리 생성
	namedWindow("img1의 키포인트");
	namedWindow("img2의 키포인트");
	namedWindow("매칭 결과");

	//*****************1.검출 with sift (SiftDescriptorExtractor)
	SIFT instance_FeatureDetector;//검출을 위한 인스턴스 생성
	std::vector<KeyPoint> img1keypoint, img2keypoint;
	instance_FeatureDetector.detect(img1, img1keypoint);//img1에 특징점을 img1keypoint에 저장
	instance_FeatureDetector.detect(img1, img2keypoint);//img1에 특징점을 img1keypoint에 저장
	//keypoint 검출 결과를 보이기
	Mat displayOfImg1, displayOfImg2;
	drawKeypoints(img1, img1keypoint, displayOfImg1, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//빨간색 DRAW_RICH_KEYPOINTS로 나타냄
	drawKeypoints(img2, img2keypoint, displayOfImg2, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//빨간색 DRAW_RICH_KEYPOINTS로 나타냄

	imshow("img1의 키포인트", displayOfImg1);
	imshow("img2의 키포인트", displayOfImg2);

	//*****************1.기술 with sift (SiftDescriptorExtractor)
	SIFT instance_Descriptor;
	Mat img1outputarray, img2outputarray;
	instance_Descriptor.compute(img1, img1keypoint, img1outputarray);
	instance_Descriptor.compute(img2, img2keypoint, img2outputarray);

	//*****************1.매칭 with sift (SiftDescriptorExtractor)
	FlannBasedMatcher FLANNmatcher;
	std::vector<DMatch> match;
	FLANNmatcher.match(img1outputarray, img2outputarray, match);
	if (!(match.size()))
	{
		std::cout << "키포인트 매칭 불가!" << std::endl;
		return -1;
	}

	//매칭된 쌍들 중에서 유클리드 거리를 기준으로 굿매치(믿을만한 매칭)을 추출
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
	//good_match인 match쌍들을 보이기
	drawMatches(img1, img1keypoint, img2, img2keypoint, good_match, finalOutputImg, Scalar(150, 30, 200), Scalar(0, 0, 255), std::vector< char >(), DrawMatchesFlags::DEFAULT);
	imshow("매칭 결과", finalOutputImg);
	waitKey(0);
	return 0;
}
