#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

TESTID=$(basename --suffix .sh -- $0)
DAXTYPE="$(choose_dax_type)"
DESC=$TESTID-$DAXTYPE

# parameters for MCAS server and client
NODE_IP="$(node_ip)"
DEBUG=${DEBUG:-0}

# parameters for MCAS server
SERVER_CONFIG="hstore-$DAXTYPE-ado0"

# launch MCAS server
DEBUG=4 DAX_RESET=1 ./dist/bin/mcas --config "$("./dist/testing/$SERVER_CONFIG.py" "$NODE_IP")" --forced-exit --debug $DEBUG &> test$TESTID-server.log &
SERVER_PID=$!

# give time to start server
sleep 3

CLIENT_LOG="test$TESTID-client.log"

# launch client
DEBUG=4 ./dist/bin/ado-test --src_addr "$NODE_IP" --server $NODE_IP --debug $DEBUG &> $CLIENT_LOG &
CLIENT_PID=$!

# arm cleanup
trap "kill -9 $SERVER_PID $CLIENT_PID &> /dev/null" EXIT

# wait for server to complete
wait $CLIENT_PID; CLIENT_RC=$?
wait $SERVER_PID; SERVER_RC=$?

pass_fail_by_code client $CLIENT_RC server $CLIENT_RC && pass_fail $CLIENT_LOG $TESTID $DESC