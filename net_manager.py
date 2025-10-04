# net_manager.py

from netmiko import ConnectHandler
from netmiko.exceptions import NetmikoTimeoutException, NetmikoAuthenticationException

def connect_to_device(device_type, host, username, password):
    """
    Establishes an SSH connection to a network device.
    """
    device = {
        'device_type': device_type,
        'host': host,
        'username': username,
        'password': password,
    }

    try:
        net_connect = ConnectHandler(**device)
        return net_connect
    except (NetmikoTimeoutException, NetmikoAuthenticationException) as e:
        print(f"Failed to connect to {host}: {e}")
        return None
    except ValueError as e:
        # Catches unsupported device_type errors from Netmiko
        print(f"Connection error: {e}")
        return None

def disconnect_from_device(connection):
    """
    Disconnects from the device.
    """
    if connection:
        connection.disconnect()
        print("Disconnected from the device.")

# These functions are no longer specific to H3C, so we remove them from here
# and handle the logic directly in the GUI's push_ai_commands method.
