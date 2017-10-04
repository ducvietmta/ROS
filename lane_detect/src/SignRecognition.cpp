#include "SignRecognition.h"
#include "fstream"
#include "math.h"

SignRecognition::SignRecognize::SignRecognize() {
	/// 1. Khởi tạo class để xử lý Descriptor của biển báo.
	signProc.loadDict("/home/nvidia/catkin_ws/src/imageview/Data/Dict.txt");

	/// 2. Đặt giá trị ompError cho xử lý.
	signProc.setOmpError(OMP_ERROR);

	/// 3. Load vector v của các biển báo trong thư viện.
	readVDictVector("/home/nvidia/catkin_ws/src/imageview/Data/VectorVDict.txt", vDictVector);

	/// 4. Load vector l dùng cho xử lý biển.
	readLVector("/home/nvidia/catkin_ws/src/imageview/Data/VectorL.txt", lVector);

	std::cerr << "ok init" << '\n';
	return ;
}

SignRecognition::SignRecognize::~SignRecognize() {
	return ;
}

void SignRecognition::SignRecognize::readVDictVector(char filename[], std::vector<Eigen::VectorXd> &v) {
	std::ifstream ifs;
	ifs.open(filename, std::ifstream::in);
	int n, m;
	ifs >> n >> m;
	std::cerr << n << " " << m << '\n';
	v.resize(n);
	for (int i = 0; i < n; i++) {
		v[i].resize(m);
		for (int j = 0; j < m; j++) ifs >> v[i](j);	
	}	
	ifs.close();
}
void SignRecognition::SignRecognize::readLVector(char fileName[], Eigen::VectorXd &l) {
	std::ifstream ifs;
	ifs.open(fileName, std::ifstream::in);
	int n, m;
	ifs >> n >> m;
	l.resize(m);
	for (int i = 0; i < m; i++) ifs >> l(i);
	ifs.close();
}
void SignRecognition::SignRecognize::makeDes(cv::Mat &des, Eigen::MatrixXd &imgDes) {
	int n = des.rows;
	int m = des.cols;
	double tmp, sum;

	imgDes.resize(n,m);

	for (int i = 0; i < n; i++) {
		sum = 0;
		for (int j = 0; j < m; j++) {
			tmp = des.at<float>(i,j);
			sum += tmp * tmp;
		}
		sum = sqrt(sum);
		for (int j = 0; j < m; j++) {
			tmp = des.at<float>(i,j);
			imgDes(i,j) = tmp / sum; 
		}
	}
	imgDes.transposeInPlace();
}
double SignRecognition::SignRecognize::cosineDistance(Eigen::VectorXd &u, Eigen::VectorXd &v) {
	return 1.0 - (u.dot(v) / (u.squaredNorm() * v.squaredNorm()));
}

void SignRecognition::SignRecognize::calcOmegaRequest(Eigen::SparseMatrix<double> &spa, Eigen::VectorXd &omega, int numCol) {
	omega.resize(numCol);
	for (int i = 0; i < numCol; i++) omega(i) = 0;
	for (int i = 0; i < spa.outerSize(); i++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(spa, i); it; ++it) {
			
			omega(it.row()) += it.value();
		}

	}
	double vmax = -1;
	for (int i = 0; i < numCol; i++) vmax = std::max(vmax, fabs(omega(i)));
	for (int i = 0; i < numCol; i++) omega(i) /= vmax;

}

void SignRecognition::SignRecognize::calcVRequest(Eigen::VectorXd &omega, Eigen::VectorXd &L, Eigen::VectorXd &v) {
	int m = omega.size();
	v.resize(m);
	for (int i = 0; i < m; i++)
		v(i) = omega(i) * L(i);
}

bool SignRecognition::SignRecognize::getSiftDescriptor(cv::Mat &sign, cv::Mat &signDes) {
	siftDes = cv::xfeatures2d::SIFT::create(0,5,0.03,10,1.3);
	cvtColor(sign, graySign, CV_BGR2GRAY);
	kp.clear();
	siftDes->detect(graySign, kp);
	if (kp.empty()) return false;
	siftDes->compute(graySign, kp, signDes);
	return true;
}
int SignRecognition::SignRecognize::recognizeSign(cv::Mat &sign) {
	/// 1. Tính SIFT descriptor của biển cần nhận dạng.
	if (getSiftDescriptor(sign, siftDesOfSign) == false) return -1;
		// Không có key point return -1
	
	/// 2. Chuẩn hóa các descriptor
	makeDes(siftDesOfSign, signDes);

	/// 3. Tính sparse matrix biểu diễn biển báo bằng dictionary
	signProc.getSparseMat(signDes);

	/// 4. Tính vector omega của biển từ sparse matrix
	calcOmegaRequest(signProc.spaMat, omegaR, signProc.dictCol);
	
	/// 5. Tính vector v biểu diễn biển báo cần nhận dạng.
	calcVRequest(omegaR, lVector, vReq);

	/// 6. Đi so sánh vector v của biển cần nhận dạng với các vector của các biển có trong thư viện.
	int numImg = vDictVector.size();
	int ret = -1;
	double mindist = 1e9, curdist;
	
	for (int i = 0; i < numImg; i++) {
		if ((i / 5 + 1) % 2 == 1) continue;
		curdist = cosineDistance(vReq, vDictVector[i]);
		if (mindist > curdist) {
			mindist = curdist;
			ret = i;
		}
	}
	if (mindist > 0.7) return -1;
	return (ret / 5) + 1;
}
