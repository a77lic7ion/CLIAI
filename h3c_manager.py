from netmiko import ConnectHandler
from netmiko.exceptions import NetmikoTimeoutException, NetmikoAuthenticationException

def connect_to_device(host, username, password):
    """
    Establishes an SSH connection to an H3C device.
    """
    device = {
        'device_type': 'hp_comware',
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

def send_config_commands(connection, commands):
    """
    Sends a list of configuration commands to the device.
    """
    if connection:
        try:
            output = connection.send_config_set(commands)
            return output
        except Exception as e:
            return f"An error occurred: {e}"
    return "Not connected to a device."

def disconnect_from_device(connection):
    """
    Disconnects from the device.
    """
    if connection:
        connection.disconnect()
        print("Disconnected from the device.")

