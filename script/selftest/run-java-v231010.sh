#!/bin/sh
### BEGIN INIT INFO
# Provides: zd yf
# Required-Start: $local_fs $syslog
# Required-Stop: $local_fs $syslog
# Default-Start: 2 3 4 5
# Default-Stop: 0 1 6
# Description: start run-java using start-stop-daemon
# Short-Description: start run-java
# 增加了结束任务循环判断逻辑
# Version: 20231010-ALPHA
### END INIT INFO

#工作目录
SHELL_FOLDER=$(dirname $(readlink -f $0))
cd $SHELL_FOLDER
APP_NAME='docker-ceping-web-240417'
INSTANCE_NAME="${APP_NAME}-1"
GREP_KEY="Dmyname=$INSTANCE_NAME"
#程序目录
APP_FOLDER="${SHELL_FOLDER}"
OUT_FILE="${APP_FOLDER}/OUT"

#主程序包
JAR_FILE="$APP_FOLDER/$APP_NAME-24.2.2-ALPHA.jar"

JAVA_HOME='/workspace/script/selftest/jdk-11.0.18'

SPRING_PROFILE0='prod'


#需要远程调试时增加参数，注意端口别冲突
REMOTE_DEBUG=''
#REMOTE_DEBUG="-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5081"
#REMOTE_DEBUG="-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=*:5081"

run_java(){
  nohup $JAVA_HOME/bin/java \
  -Xms200m -Xmx500m -XX:MetaspaceSize=150m \
  -$GREP_KEY \
  -Duser.timezone=Asia/Shanghai \
  -Dfile.encoding=utf-8 \
  $REMOTE_DEBUG \
  -jar $JAR_FILE \
  --spring.profiles.active=$1 \
  1>$OUT_FILE 2>&1 &
}

start0(){
  is_exist
  if [ $? -eq 0 ];then
    echo "$INSTANCE_NAME is already running. pid=$PID."
    return 0
  else
    run_java $SPRING_PROFILE0
    sleep 3
    is_exist
    if [ $? -eq 0 ];then
      echo "$INSTANCE_NAME is running now. pid=$PID."
    else
      echo "$INSTANCE_NAME run failed."
    fi
  fi
  rm /workspace/zd_pipe.txt
  touch /workspace/zd_pipe.txt
}

is_exist(){
  PID=`ps -ef|grep -v "grep" |grep -w $GREP_KEY|awk '{print $2}'`
  if [ -z "$PID" ];then
    return 1
  else
    return 0
  fi
}

status(){
  is_exist
  if [ $? -eq 0 ];then
    echo "$INSTANCE_NAME is running. pid=$PID."
  else
    echo "$INSTANCE_NAME is not running."
  fi
}

stop0(){
  is_exist
  if [ $? -eq 0 ];then
    kill -15 ${PID}
    if [ $? -ne 0 ];then
      echo "stop $INSTANCE_NAME failed!"
      return 1
    else
      is_exist
      while [ $? -eq 0 ]
      do
        sleep 1
        echo '.'
        is_exist
      done	
      echo "$INSTANCE_NAME stopped."
      return 0
    fi
  else
    echo "$INSTANCE_NAME is not running!"
    return 1
  fi
}

restart(){
  stop0
  sleep 3
  start0
}

usage(){
  echo "usage: $0 {start|restart|stop|status}"
}

case "$1" in
  'start')
    start0
    ;;
  'stop')
    stop0
    ;;
  'restart')
    restart
    ;;
  'status')
    status
    ;;
  *)
    usage
    ;;
esac
