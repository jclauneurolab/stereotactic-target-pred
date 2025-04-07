import argparse
from pathlib import Path, PurePath
from fabric import Connection, config


def push_release(
    connection: Connection,
    new_release_path: Path,
    remote_releases_dir: PurePath,
    wheel_name: str,
):
    remote_release_new = str(remote_releases_dir / new_release_path.name)
    connection.run(f"mkdir -p {remote_release_new}")
    connection.put(str(new_release_path / wheel_name), remote_release_new)
    
    env_path = new_release_path / ".env"
    if env_path.exists():
        connection.put(str(env_path), remote_release_new)

    link_name = str(remote_releases_dir.parent / "current")
    connection.run(f"ln -sfn {remote_release_new} {link_name}")


def install_wheel(connection: Connection, venv_path: str, wheel_path: str):
    connection.run(
        f"source {venv_path}/bin/activate && pip install {wheel_path}"
    )

def restart_service(connection: Connection, service_name: str):
    connection.sudo(f"systemctl restart {service_name}")

def deploy(
    connection: Connection,
    new_release_path: Path,
    remote_releases_dir: PurePath,
    wheel_name: str,
    venv_path: PurePath,
    service_name: str,
):
    push_release(connection, new_release_path, remote_releases_dir, wheel_name)
    wheel_path = str(remote_releases_dir / new_release_path.name / wheel_name)
    install_wheel(connection, str(venv_path), wheel_path)
    restart_service(connection, service_name)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("new_release_path", type=Path)
    parser.add_argument("remote_releases_dir", type=PurePath)
    parser.add_argument("wheel_name")
    parser.add_argument("venv_path", type=PurePath)
    parser.add_argument("identity_path", type=Path)
    parser.add_argument("service_name")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    conn = Connection(
        args.host, connect_kwargs={"key_filename": str(args.identity_path)}
    )
    deploy(
        conn,
        args.new_release_path,
        args.remote_releases_dir,
        args.wheel_name,
        args.venv_path,
        args.service_name,
    )

if __name__ == "__main__":
    config.Config()
    main()
