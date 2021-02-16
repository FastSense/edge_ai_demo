#!/bin/bash

command="exec"
build_services="base rs rs_ov rs_ov_tf"

terminal="bash"
exec_services="rs_ov_tf"


if [ -n "$1" ] 
  then command=$1 
fi

if [ -n "$2" ] 
  then build_services=$2 
fi

case $command in
exec)
  docker-compose exec $exec_services $terminal ;;

up)
  docker-compose up -d $exec_services ;;

stop)
  docker-compose stop $exec_services ;;

build)
  services=(${build_services//' '/ })
  for service in "${services[@]}"
  do
    docker-compose build ${service}
  done ;;

*)
  echo "First arg possible values: exec, up, build, stop"
  echo "Second arg possible values: build services..."
  echo "Current exec service: $exec_services"
;;
esac
