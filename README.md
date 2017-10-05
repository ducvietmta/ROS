# ROS
learn and test ROS
Các lệnh dùng với ROS package:
catkin_create_pkg: tạo package mới (nằm trong catkin_ws/src)
rospack: lấy info của các package
catkin_make: build workspace của package
rosdep: cài dependencies của package
roscd (ros change dir): đổi đường dẫn của 1 package 
roscp: copy 1 file
rosed: edit 1 file
rosrun: chạy executables

Các lệnh dùng với ROS node:
rosnode info [node_name]: in info về node
rosnode kill [node_name]: kill node
rosnode list: list các node
rosnode machine [machine_name]: list các node của 1 machine
rosnode ping: kiểm tra đường truyền đến node
rosnode cleanup: xoá các node ko kết nối đến dc

Các lệnh dùng với ROS topic:
rostopic bw /topic: hiện bandwidth của 1 topic
rostopic echo /topic: hiện nội dung của 1 topic
rostopic find /message_type: tìm topic với loại message nhất định
rostopic hz /topic: hiện tốc độ publish (Hz)
rostopic info /topic: in info về 1 topic đang hoạt động
rostopic list: list tất cả topic đang hoạt động
rostopic pub /topic message_type args: publish 1 giá trị cho 1 topic với 1 loại message
rostopic type /topic: hiện loại message của 1 topic

Các lệnh dùng với ROS service:
rosservice call /service args: gọi service với những argument nhất định
rosservice find service_type: tìm service với 1 loại service nhất định
rosservice info /services: in info về 1 service
rosservice list: list các service đang hoạt động
rosservice type /service: in loại service của 1 service
rosservice uri /service: in ROSRPC URI của service

Các lệnh dùng với ROS bag:
rosbag record [topic_1] [topic_2] -o [bag_name]: ghi lại dữ liệu từ các topic vào 1 file bag. (có thể dùng với “-a” thay vì “-o”)
rosbag play [bag_name]: chơi lại dữ liệu trên 1 file bag.
*rqt_bag

Các lệnh dùng với ROS parameter:
rosparam set [parameter_name] [value]: đặt giá trị cho 1 tham số
rosparam get [parameter_name]: hiện giá trị của 1 tham số
rosparam load   [YAML file]: tải ra các parameter từ 1 file YAML
rosparam dump [YAML file]: viết các parameter vào 1 file YAML
rosparam delete [parameter_name]: xoá parameter
rosparam list: list các parameter
