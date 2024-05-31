# 自助测试

> 主要测试内容：接口返回值格式、算法稳定性

* 获取该项目到本地工作目录下并赋予文件所需权限

    ```bash
    cd /tmp
    ```
    ```bash
    git clone https://github.com/nelivacn/FAT-selfTest.git
    ```
    ```bash
    mkdir -p /workspace/projects/ /workspace/tars/
    ```
    ```bash
    cp -r /tmp/FAT-selfTest/script/ /workspace/
    ```
    ```bash
    chmod -R 777 /workspace/script/
    ```

* 修改 **docker.service** 文件 配置TCP远程访问

    ```bash
    systemctl stop docker
    ```
    ```bash
    vim /lib/systemd/system/docker.service
    ```
    ```bash
    # [Service] ExecStart 项新增  -H tcp://0.0.0.0:2345
    # 例 ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock -H tcp://0.0.0.0:2345
    # 保存退出
    ```
    ```bash
    systemctl daemon-reload
    ```
    ```bash
    systemctl start docker
    ```

* 获取镜像
    **selftest** 版本包含测试图片等依赖 正式测评使用 **base** 版本镜像

    ```bash
    docker pull nelivacn/fat:cuda12.2.2-ubuntu22.04-selftestV2024.2
    ```

    ```bash
    docker pull nelivacn/fat:cuda11.4.3-ubuntu18.04-selftestV2024.2
    ```

    ```bash
    docker pull nelivacn/fat:cuda12.2.2-centos7-selftestV2024.2
    ```

    ```bash
    docker pull nelivacn/fat:cuda11.4.3-centos7-selftestV2024.2
    ```

* 检验容器环境

    ```bash
    docker run -d --privileged --gpus all --ipc host --network host --name imageTest nelivacn/fat:cuda11.4.3-ubuntu18.04-selftestV2024.1
    ```

    ```bash
    docker exec -it imageTest /bin/bash
    ```

    执行一些命令查看容器环境是否正常
    ```bash
    nvidia-smi && nvcc -V
    ```

    退出并关闭删除容器
    ```bash
    exit
    ```
    
    ```bash
    docker stop imageTest
    ```
    
    ```bash
    docker rm imageTest
    ```

* 将需要测试的程序包上传至服务器 **/workspace/tars/** 目录下

* 启动测试服务

    ```bash
    cd /workspace/script/selftest/
    ```
    ```bash
    ./selfTest.sh start
    ```

* 开始自助测试

    1. 进入测试页面[http://ip:8040/self/test](http://127.0.0.1:8040/self/test)
    2. 输入**镜像名称**与**程序包所在绝对路径**
    3. 点击**初始化**按钮并观察测试日志输出
    4. 如果需要授权请点击**下载**按钮下载指纹文件、**上传**按钮上传授权文件进行授权操作
    5. 等待初始化完成后，点击**功能测试**开始进行自助测试
    6. 如果当前状态为**测试通过**，请点击**下载测试日志**测试日志文件同程序包一同提交
    7. 如果当前状态为**测试失败**，请在日志输出区域中查找失败原因并更正后重新进行自助测试

    **浏览器不要刷新**

    **重新测试需要重启测试服务./selfTest.sh restart**

    **该自助测试只涉及接口格式校验以及稳定性测试，不涉及任何性能测试**

* 技术相关的问题请在[Issues](https://github.com/nelivacn/FAT-selfTest/issues)进行提问讨论
