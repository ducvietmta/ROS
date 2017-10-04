#include "SignDetection.h"

SignDetection::Detection::Detection() {
    return ;
}

SignDetection::Detection::~Detection() {
    return ;
}

void SignDetection::Detection::BGR2HSV(double b, double g, double r, double &h, double &s, double &v) {
    double mnv, mxv, delta;

    mnv = mxv = r;
    if (mnv > g) mnv = g;
    if (mnv > b) mnv = b;
    if (mxv < g) mxv = g;
    if (mxv < b) mxv = b;
    v = mxv;

    delta = mxv - mnv;

    if ( mxv != 0.0)
        s = delta / mxv;
    else {
        // r = g = b = 0		// s = 0, v is undefined
        s = 0.0;
        h = -1.0;
        return;
    }

    if (std::fabs(r - mxv) < 1e-6)
        h = (g - b ) / delta;		// between yellow & magenta
    else if (std::fabs(g - mxv) < 1e-6)
        h = 2 + ( b - r ) / delta;	// between cyan & yellow
    else
        h = 4 + ( r - g ) / delta;	// between magenta & cyan

    h *= 60;				// degrees
    if (h < 0.0)
        h += 360;
    h /= 360.0;
}

void SignDetection::Detection::BGR2HSV(const cv::Mat &bgr, cv::Mat &hsv) {
    hsv = cv::Mat::zeros(bgr.rows, bgr.cols, CV_32FC3);
    for (int i = 0; i < bgr.rows; ++i)
        for (int j = 0; j < bgr.cols; ++j) {
            double b = (double)bgr.at<cv::Vec3b>(i, j)[0];
            double g = (double)bgr.at<cv::Vec3b>(i, j)[1];
            double r = (double)bgr.at<cv::Vec3b>(i, j)[2];
            double h, s, v;
            this->BGR2HSV(b, g, r, h, s, v);
            hsv.at<cv::Vec3f>(i, j)[0] = h;
            hsv.at<cv::Vec3f>(i, j)[1] = s;
            hsv.at<cv::Vec3f>(i, j)[2] = v;
        }
}

bool SignDetection::Detection::inside(int u, int v, int n, int m) {
    return (0 <= u && u < n && 0 <= v && v < m);
}

void SignDetection::Detection::findComponents(const cv::Mat &binary, std::vector<cv::Rect> &boxes, double lower_ratio, double upper_ratio, int lower_numpoint) {
    boxes.clear();
    int n = binary.rows;
    int m = binary.cols;
    const static int du[] = {0, 0, 1, -1};
    const static int dv[] = {1, -1, 0, 0};
    char **mark = new char*[n];
    //'.': not visited
    //'x': visited, not satisfied
    //'o': visited and satisfied
    for(int r = 0; r < n; ++r) {
        mark[r] = new char[m];
        for(int c = 0; c < m; ++c)
            mark[r][c] = '.';
    }
    size_t largest = 0;
    std::vector< std::vector<cv::Point2i> > regions;
    std::queue<int> q;
    for(int r = 0; r < n; ++r)
        for(int c = 0; c < m; ++c)
            if (binary.at<uchar>(r, c) != 0 && mark[r][c] == '.'){
                int minu = (int)1e9, maxu = -1;
                int minv = (int)1e9, maxv = -1;
                q.push(r);
                q.push(c);
                std::vector<cv::Point2i> region;
                while (!q.empty()) {
                    int u = q.front(); q.pop();
                    int v = q.front(); q.pop();
                    for(int dir = 0; dir < 4; ++dir) {
                        int nu = u + du[dir];
                        int nv = v + dv[dir];
                        if (nu < 0 || nu >= n || nv < 0 || nv >= m) continue;
                        if (mark[nu][nv] != '.') continue;
                        if (binary.at<uchar>(nu, nv) == 0) {
                            mark[nu][nv] = 'x';
                            continue;
                        }
                        int ones = 0;
                        int vals = 0;
                        for(int dd = 0; dd < 4; ++dd)
                            if (nu + du[dd] >= 0 && nu + du[dd] < n && nv + dv[dd] >= 0 && nv + dv[dd] < m) {
                                ++vals;
                                if (binary.at<uchar>(nu + du[dd], nv + dv[dd]) != 0) ++ones;
                            }
                        if (abs(ones-vals) < 3){// || (ones == 3 && vals == 4)) {
                            if (minu > nu) minu = nu;
                            if (maxu < nu) maxu = nu;
                            if (minv > nv) minv = nv;
                            if (maxv < nv) maxv = nv;
                            mark[nu][nv] = 'o';
                            region.push_back(cv::Point2i(nu, nv));
                            q.push(nu);
                            q.push(nv);
                        }
                        else
                            mark[nu][nv] = 'x';
                    }
                }
                if (maxu - minu < 10 || maxv - minv < 10) continue;
                if ((int)region.size() < lower_numpoint) continue;
                double ratio = (1.0 * maxu - minu) / (maxv - minv);
                if (ratio < lower_ratio || ratio > upper_ratio) continue;
                boxes.push_back(cv::Rect(minv, minu, maxv - minv, maxu - minu));
                regions.push_back(region);
                if (largest < region.size())
                    largest = region.size();
            }
    for(size_t i = 0; i < regions.size(); ++i)
        if (4 * regions[i].size() < largest) {
            std::swap(regions[i], regions.back());
            std::swap(boxes[i], boxes.back());
            regions.pop_back();
            boxes.pop_back();
        }
    for(int i = 0; i < n; ++i)
        delete[] mark[i];
    delete mark;
}

void SignDetection::Detection::redToBinary(cv::Mat &result, const cv::Mat &bgr) {
    result = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8U);
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    
    /*
    for(int r = 0; r < hsv.rows; ++r)
    for(int c = 0; c < hsv.cols; ++c)
        if (hsv.at<cv::Vec3f>(r, c)[0] < 0.05 || hsv.at<cv::Vec3f>(r, c)[0] > 0.95)
        if (hsv.at<cv::Vec3f>(r, c)[1] > 0.4 && hsv.at<cv::Vec3f>(r, c)[2] > 0.15)
            result.at<uchar>(r, c) = 255;
    */
    inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), result);
 
}

void SignDetection::Detection::blueToBinary(cv::Mat &result, const cv::Mat &bgr)
{
	result = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8U);
	cv::Mat hsv;
	this->BGR2HSV(bgr, hsv);
	for (int r = 0; r < hsv.rows; r++)
	for (int c = 0; c < hsv.cols; c++)
		if (hsv.at<cv::Vec3f>(r,c)[0] > 1.0 / 6 && hsv.at<cv::Vec3f>(r,c)[0] < 0.7 &&
		    hsv.at<cv::Vec3f>(r,c)[1] > 0.4 && hsv.at<cv::Vec3f>(r,c)[2] > 0.4314)
			result.at<uchar>(r,c) = 255;
}

bool SignDetection::Detection::floodFill(std::vector<cv::Point> &res, cv::Mat &image, cv::Point &p, ushort connected_condition) {
    ushort hole = 0;
    ushort tmp, current_val;
    cv::Point n;
    cv::Point new_point;
    double neighbour = 0;
    double connected = 0;
    bool is_connected = true;
    res.clear();

    if (image.at<ushort>(p) == hole) return false;
    std::vector<cv::Point> q;
    q.push_back(p);
    while (q.size() != 0) {
        neighbour = 0;
        connected = 0;
        n = q[0];
        q.erase(q.begin());
        if (image.at<ushort>(n) != hole) {
            current_val = image.at<ushort>(n);
            image.at<ushort>(n) = hole;


            new_point.x = n.x + 1;
            if (new_point.x <image.cols) {
                new_point.y = n.y;
                tmp = image.at<ushort>(new_point);
                neighbour++;
                if ((tmp >= (current_val-connected_condition))&&(tmp <= (current_val+connected_condition))) {
                    q.push_back(new_point);
                    connected++;
                }
            }

            new_point.x = n.x - 1;
            if (new_point.x >= 0) {
                new_point.y = n.y;
                tmp = image.at<ushort>(new_point);
                neighbour++;
                if ((tmp >= (current_val-connected_condition))&&(tmp <= (current_val+connected_condition))) {
                    q.push_back(new_point);
                    connected++;
                }
            }

            new_point.y = n.y + 1;
            if (new_point.y <image.rows) {
                new_point.x = n.x;
                tmp = image.at<ushort>(new_point);
                neighbour++;
                if ((tmp >= (current_val-connected_condition))&&(tmp <= (current_val+connected_condition))) {
                    q.push_back(new_point);
                    connected++;
                }
            }

            new_point.y = n.y - 1;
            if (new_point.y >= 0) {
                new_point.x = n.x;
                tmp = image.at<ushort>(new_point);
                neighbour++;
                if ((tmp >= (current_val-connected_condition))&&(tmp <= (current_val+connected_condition))) {
                    q.push_back(new_point);
                    connected++;
                }
            }

            new_point.x = n.x + 1;
            new_point.y = n.y + 1;
            if ((new_point.x < image.cols) && (new_point.y < image.rows)) {
                tmp = image.at<ushort>(new_point);
                neighbour++;
                if ((tmp >= (current_val-connected_condition))&&(tmp <= (current_val+connected_condition))) {
                    q.push_back(new_point);
                    connected++;
                }
            }

            new_point.x = n.x + 1;
            new_point.y = n.y - 1;
            if ((new_point.x < image.cols) && (new_point.y >= 0)) {
                tmp = image.at<ushort>(new_point);
                neighbour++;
                if ((tmp >= (current_val-connected_condition))&&(tmp <= (current_val+connected_condition))) {
                    q.push_back(new_point);
                    connected++;
                }
            }

            new_point.x = n.x - 1;
            new_point.y = n.y + 1;
            if ((new_point.x >= 0) && (new_point.y < image.rows)) {
                tmp = image.at<ushort>(new_point);
                neighbour++;
                if ((tmp >= (current_val-connected_condition))&&(tmp <= (current_val+connected_condition))) {
                    q.push_back(new_point);
                    connected++;
                }
            }

            new_point.x = n.x - 1;
            new_point.y = n.y - 1;
            if ((new_point.x >= 0) && (new_point.y >= 0)) {
                tmp = image.at<ushort>(new_point);
                neighbour++;
                if ((tmp >= (current_val-connected_condition))&&(tmp <= (current_val+connected_condition))) {
                    q.push_back(new_point);
                    connected++;
                }
            }

            if (connected/neighbour >= 0.5) res.push_back(n);
        }
    }
    return true;
}
/*
bool SignDetection::Detection::objectLabeling(std::vector<std::vector<cv::Point> > &regs, cv::Mat &depth, ushort &low_th, ushort &high_th) {
    ushort depth_val;
    ushort hole = 0, tmp_v;
    cv::Point p;
    std::vector<cv::Point> reg;

    cv::Mat clone_depth = depth.clone();

    for (int y = 0; y < clone_depth.rows; y++) {
        for (int x = 0; x < clone_depth.cols; x++) {
            depth_val = clone_depth.at<ushort>(y,x);
            if ((depth_val < low_th) || (depth_val > high_th)) clone_depth.at<ushort>(y,x) = 0;
        }
    }

    regs.clear();
    reg.push_back(cv::Point(0,0));
    for (int y = 1; y < clone_depth.rows; y++) {
        for (int x = 1; x < clone_depth.cols -1; x++) {
            p.x = x;
            p.y = y;
            tmp_v = clone_depth.at<ushort>(p);
            if (tmp_v != hole) {
                this->floodFill(reg, clone_depth, p, 7);
                regs.push_back(reg);
            }
        }
    }
    regs.shrink_to_fit();

    return true;
}
*/
bool SignDetection::Detection::objectLabeling(std::vector<cv::Rect> &boxes, std::vector<int> &labels, cv::Mat &depth, cv::Mat &color, ushort &low_th, ushort &high_th, int min_pts, int max_pts, ushort min_w, ushort max_w, double ratio)
{
	ushort depth_val;
	ushort hole = 0, tmp_v;
	cv::Point p;
	std::vector<cv::Point> reg;
	
	cv::imshow("xcolor", color);
	

	cv::Mat clone_depth;
	depth.copyTo(clone_depth);
	cv::imshow("xdepth", clone_depth);
	cv::waitKey();	
	for (int y = 0; y < clone_depth.rows; y++)
	{
		for (int x = 0; x < clone_depth.cols; x++)
		{
			depth_val = clone_depth.at<ushort>(y,x);
			if ((depth_val < low_th) || (depth_val > high_th)) clone_depth.at<ushort>(y,x) = 0;
		}
	}
	std::cerr << "xxxxx" << '\n';
	boxes.clear();
	reg.push_back(cv::Point(0,0));
	
	for (int y = 1; y < clone_depth.rows; y++)
	{
		for (int x = 1; x < clone_depth.cols - 1; x++)
		{
			p.x = x;
			p.y = y;
			tmp_v = clone_depth.at<ushort>(p);
			if (tmp_v != hole)
			{
				this -> floodFill(reg, clone_depth, p, 7);
				if (reg.size() < min_pts || reg.size() > max_pts) continue;
				cv::Point p1, p2;
				p1 = p2 = reg[0];
				for (int i = 0; i < reg.size(); i++)
				{
					if (p1.x > reg[i].x) p1.x = reg[i].x;
					if (p1.y > reg[i].y) p1.y = reg[i].y;
					if (p2.x < reg[i].x) p2.x = reg[i].x;
					if (p2.y < reg[i].y) p2.y = reg[i].y;
				}
				int w = p2.x - p1.x;
				int h = p2.y - p1.y;
				if (h < min_w || h > max_w || w < min_w || w > max_w || float(h) / w > ratio || float(w) / h > ratio) continue;
				if (p1.x - 20 >= 0) p1.x -= 20;
				if (p1.y - 5 >= 0) p1.y -= 5;
				if (p2.x + 5 < clone_depth.cols) p2.x += 5;
				if (p2.y + 5 < clone_depth.rows) p2.y += 5;
				cv::Mat object = color(cv::Rect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y));
				cv::Mat binary;
				this->redToBinary(binary, object);
				std::vector<cv::Rect> colorRegions;
				this->findComponents(binary, colorRegions);
				if (colorRegions.size() > 0)
				{
					int index = 0;
					int sMax = colorRegions[index].width * colorRegions[index].height;
					for (int i = 1; i < colorRegions.size(); i++)
						if (sMax < colorRegions[i].width * colorRegions[i].height)
						{
							sMax = colorRegions[i].width * colorRegions[i].height;
							index = i;
						}
					int pad = 5, top = 0, right = 0, bot = 0, left = 0;
					colorRegions[index].x += p1.x;
					colorRegions[index].y += p1.y;
					if (colorRegions[index].x - pad >= 0) left = pad;
					if (colorRegions[index].y - pad >= 0) top = pad;
					if (colorRegions[index].x + colorRegions[index].width + pad < color.cols) right = pad;
					if (colorRegions[index].y + colorRegions[index].height + pad < color.rows) bot = pad;
					colorRegions[index].x -= left;
					colorRegions[index].y -= top;
					colorRegions[index].width += left + right;
					colorRegions[index].height += top + bot;
					boxes.push_back(colorRegions[index]);
					labels.push_back(0);
				} else
				{
					this->blueToBinary(binary, object);
					this->findComponents(binary, colorRegions);
					if (colorRegions.size() > 0)
					{
						int index = 0;
						int sMax = colorRegions[index].width * colorRegions[index].height;
						for (int i = 1; i < colorRegions.size(); i++)
							if (sMax < colorRegions[i].width * colorRegions[i].height)
							{
								sMax = colorRegions[i].width * colorRegions[i].height;
								index = i;
							}
						int pad = 5, top = 0, right = 0, bot = 0, left = 0;
						colorRegions[index].x += p1.x;
						colorRegions[index].y += p1.y;
						if (colorRegions[index].x - pad >= 0) left = pad;
						if (colorRegions[index].y - pad >= 0) top = pad;
						if (colorRegions[index].x + colorRegions[index].width + pad < color.cols) right = pad;
						if (colorRegions[index].y + colorRegions[index].height + pad < color.rows) bot = pad;
						colorRegions[index].x -= left;
						colorRegions[index].y -= top;
						colorRegions[index].width += left + right;
						colorRegions[index].height += top + bot;	
						boxes.push_back(colorRegions[index]);
						labels.push_back(1);
					}
				}
			}
		}
	}
	boxes.shrink_to_fit();
	return true;
}

bool SignDetection::Detection::saveDetectedObjects(std::vector<std::vector<cv::Point> > &regs, cv::Mat &image, std::string &path, size_t &image_index) {
    std::vector<cv::Point> reg;
    std::vector<std::vector<cv::Point> >::iterator it;
    std::vector<cv::Point>::iterator it_point;
    cv::Point p1, p2;
    size_t m_index = 0;
    for (it = regs.begin(); it != regs.end(); ++it) {
        reg = *it;
        if (reg.size() <200 || reg.size() > 2000) continue;
        p1=p2=reg[0];
        for (it_point = reg.begin(); it_point != reg.end(); ++it_point) {
            if (it_point->x < p1.x) p1.x = it_point->x;
            if (it_point->y < p1.y) p1.y = it_point->y;
            if (it_point->x > p2.x) p2.x = it_point->x;
            if (it_point->y > p2.y) p2.y = it_point->y;
        }

        if (p1.x - 5 >= 0) p1.x = p1.x -5;
        if (p1.y - 5 >= 0) p1.y = p1.y -5;
        if (p2.x + 5 < image.cols) p2.x = p2.x + 5;
        if (p2.y + 5 < image.rows) p2.y = p2.y + 5;
        uint h_ = p2.y - p1.y;
        uint w_ = p2.x - p1.x;
        if (h_ < 20 || h_ > 100 || w_ < 20 || w_ >100) continue;
        cv::rectangle(image,p1, p2, cv::Scalar(0,0,255));
        std::string file_name = path + std::to_string(image_index) + "_reg_" + std::to_string(m_index)+".jpg";

        cv::imwrite(file_name, image);
        m_index++;
    }
    return true;
}
