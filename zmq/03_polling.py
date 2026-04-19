import zmq

def simple_polling():

    print("polling")
    print("=" * 40)

    ctx = zmq.Context()

    # socket
    s1 = ctx.socket(zmq.REP)
    s2 = ctx.socket(zmq.REQ)

    s1.bind("tcp://*:7000")
    s2.connect("tcp://localhost:7000")

    # poller
    poller = zmq.Poller()
    poller.register(s1, zmq.POLLIN)

    print("setup done:")
    print("  s1: REP server (port 7000)")
    print("  s2: REQ client")
    print("  poller monitoring s1")

    print("\n[Client] Sending request...")
    s2.send_string("Hello")

    # poll 3 times
    for i in range(3):
        print(f"\nPolling {i+1}:")

        # poll wait for 1 second
        socks = dict(poller.poll(1000))

        if s1 in socks:
            print("  s1 has data!")
            msg = s1.recv_string()
            print(f"  Received: {msg}")

            # send response
            s1.send_string(f"World {i+1}")
            print(f"  Sent response: World {i+1}")

            # client receives response
            resp = s2.recv_string()
            print(f"  Client received response: {resp}")
        else:
            print("  No events")

    s1.close()
    s2.close()
    ctx.term()

if __name__ == "__main__":
    simple_polling()