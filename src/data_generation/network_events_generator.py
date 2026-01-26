"""
Module: network_events_generator.py
===================================

Generate synthetic network event log data with embedded cluster patterns
for unsupervised learning demonstration (K-means + LSA).

Creates network events with 6 distinct behavioral clusters that K-means
should discover through numerical features and text patterns in log messages.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..utils.helpers import set_seed, load_config
from ..utils.logging_config import get_logger


# Log message templates by cluster type
LOG_TEMPLATES = {
    "normal_web": [
        "GET request to {path} from {ip} completed successfully",
        "HTTPS connection established to {domain}",
        "Web page served: {path} (200 OK)",
        "Static asset loaded: {asset} from CDN",
        "API request processed: {endpoint} in {ms}ms",
        "User session started for {user_type} user",
        "Browser {browser} connected via {protocol}",
    ],
    "normal_db": [
        "SELECT query executed on {table} in {ms}ms",
        "Database connection pool: {count} active connections",
        "Query optimization: index scan on {table}",
        "Prepared statement executed: {operation}",
        "Transaction committed: {txn_id}",
        "Replica sync completed: {lag}ms lag",
        "Connection to {db_host} established",
    ],
    "suspicious_scan": [
        "Port scan detected from {ip}: {port_count} ports probed",
        "Unusual connection pattern: rapid {protocol} requests",
        "Network enumeration attempt from {ip}",
        "SYN flood pattern detected on port {port}",
        "Suspicious reconnaissance: ICMP sweep from {ip}",
        "Multiple failed connections to closed ports",
        "Rate limit exceeded: {ip} blocked temporarily",
    ],
    "auth_failure": [
        "Authentication failed for user {username}: invalid credentials",
        "Login attempt blocked: too many failures from {ip}",
        "Password retry limit exceeded for {username}",
        "SSH key authentication rejected for {username}",
        "LDAP bind failed: invalid DN or password",
        "MFA verification failed: incorrect code",
        "Account lockout triggered for {username} after {count} attempts",
    ],
    "data_exfil": [
        "Large data transfer detected: {size}MB to external {ip}",
        "Unusual outbound traffic pattern: {protocol} to {domain}",
        "Data export job completed: {size}MB transferred",
        "Bulk download initiated by {username}",
        "External API call with large payload: {size}KB",
        "FTP upload detected to non-whitelisted host",
        "DNS tunneling pattern detected to {domain}",
    ],
    "maintenance": [
        "Scheduled backup completed: {size}GB archived",
        "System update applied: {component} version {version}",
        "Log rotation completed: {count} files archived",
        "Health check passed: all services operational",
        "Cron job executed: {job_name} (exit code 0)",
        "Disk cleanup: {size}GB reclaimed",
        "Certificate renewal completed for {domain}",
    ],
}


def generate_network_events(
    config_path: str = "config/settings.yaml",
    output_path: Optional[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic network event data with embedded cluster patterns.

    Args:
        config_path: Path to configuration YAML file
        output_path: Optional path to save generated CSV
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic network events
    """
    set_seed(seed)
    logger = get_logger()

    config = load_config(config_path)
    net_config = config["data_generation"]["network_events"]

    n_events = net_config["n_events"]
    time_window_days = net_config["time_window_days"]
    cluster_types = net_config["cluster_types"]
    common_ports = net_config["common_ports"]

    logger.info(f"Generating {n_events:,} network events...")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=time_window_days)

    cluster_counts = {
        cluster: int(n_events * proportion)
        for cluster, proportion in cluster_types.items()
    }

    # Adjust for rounding
    diff = n_events - sum(cluster_counts.values())
    if diff > 0:
        cluster_counts["normal_web"] += diff

    logger.info(f"Cluster distribution: {cluster_counts}")

    all_records = []

    for cluster_name, count in cluster_counts.items():
        logger.info(f"Generating {count:,} {cluster_name} events...")
        records = _generate_cluster_events(
            cluster_name=cluster_name,
            n_events=count,
            start_time=start_time,
            end_time=end_time,
            common_ports=common_ports,
        )
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Generated {len(df):,} network events")
    logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Cluster counts:\n{df['true_cluster'].value_counts()}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to: {output_path}")

    return df


def _generate_cluster_events(
    cluster_name: str,
    n_events: int,
    start_time: datetime,
    end_time: datetime,
    common_ports: Dict[str, List[int]],
) -> List[Dict]:
    """
    Generate events for a specific cluster type.

    Args:
        cluster_name: Name of the cluster type
        n_events: Number of events to generate
        start_time: Start of time window
        end_time: End of time window
        common_ports: Dictionary of port lists by type

    Returns:
        List of event dictionaries
    """
    records = []
    time_range = (end_time - start_time).total_seconds()

    cluster_params = _get_cluster_params(cluster_name)

    for _ in range(n_events):
        # Timestamp generation (maintenance is off-hours)
        if cluster_name == "maintenance":
            hour_offset = np.random.uniform(2, 6) * 3600
            day_offset = np.random.randint(0, (end_time - start_time).days)
            timestamp = start_time + timedelta(days=day_offset, seconds=hour_offset)
        else:
            random_seconds = np.random.uniform(0, time_range)
            timestamp = start_time + timedelta(seconds=random_seconds)

        source_ip = _generate_ip(cluster_params["source_ip_type"])
        dest_ip = _generate_ip(cluster_params["dest_ip_type"])
        port = _generate_port(cluster_name, common_ports)

        # Log-normal distributions for numerical features
        duration_ms = max(1, np.random.lognormal(
            cluster_params["duration_mean"],
            cluster_params["duration_std"]
        ))

        bytes_transferred = max(0, np.random.lognormal(
            cluster_params["bytes_mean"],
            cluster_params["bytes_std"]
        ))

        protocol = np.random.choice(
            cluster_params["protocols"],
            p=cluster_params["protocol_probs"]
        )

        severity = np.random.choice(
            cluster_params["severities"],
            p=cluster_params["severity_probs"]
        )

        log_message = _generate_log_message(cluster_name)

        record = {
            "timestamp": timestamp,
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "port": port,
            "duration_ms": round(duration_ms, 2),
            "bytes_transferred": int(bytes_transferred),
            "protocol": protocol,
            "severity": severity,
            "log_message": log_message,
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "true_cluster": cluster_name,
        }
        records.append(record)

    return records


def _get_cluster_params(cluster_name: str) -> Dict:
    """
    Get characteristic parameters for each cluster type.

    Args:
        cluster_name: Cluster type name

    Returns:
        Dictionary of distribution parameters
    """
    params = {
        "normal_web": {
            "duration_mean": 5.0,
            "duration_std": 1.5,
            "bytes_mean": 9.0,
            "bytes_std": 2.0,
            "source_ip_type": "external",
            "dest_ip_type": "internal",
            "protocols": ["TCP", "UDP"],
            "protocol_probs": [0.95, 0.05],
            "severities": ["info", "warning"],
            "severity_probs": [0.95, 0.05],
        },
        "normal_db": {
            "duration_mean": 2.0,
            "duration_std": 1.0,
            "bytes_mean": 8.0,
            "bytes_std": 2.5,
            "source_ip_type": "internal",
            "dest_ip_type": "internal",
            "protocols": ["TCP"],
            "protocol_probs": [1.0],
            "severities": ["info"],
            "severity_probs": [1.0],
        },
        "suspicious_scan": {
            "duration_mean": 1.0,
            "duration_std": 0.5,
            "bytes_mean": 5.0,
            "bytes_std": 1.0,
            "source_ip_type": "external",
            "dest_ip_type": "internal",
            "protocols": ["TCP", "UDP", "ICMP"],
            "protocol_probs": [0.6, 0.2, 0.2],
            "severities": ["warning", "error", "critical"],
            "severity_probs": [0.4, 0.4, 0.2],
        },
        "auth_failure": {
            "duration_mean": 4.0,
            "duration_std": 1.0,
            "bytes_mean": 6.0,
            "bytes_std": 1.0,
            "source_ip_type": "external",
            "dest_ip_type": "internal",
            "protocols": ["TCP"],
            "protocol_probs": [1.0],
            "severities": ["warning", "error"],
            "severity_probs": [0.3, 0.7],
        },
        "data_exfil": {
            "duration_mean": 8.0,
            "duration_std": 2.0,
            "bytes_mean": 15.0,
            "bytes_std": 3.0,
            "source_ip_type": "internal",
            "dest_ip_type": "external",
            "protocols": ["TCP", "UDP"],
            "protocol_probs": [0.8, 0.2],
            "severities": ["warning", "error", "critical"],
            "severity_probs": [0.3, 0.4, 0.3],
        },
        "maintenance": {
            "duration_mean": 7.0,
            "duration_std": 2.5,
            "bytes_mean": 12.0,
            "bytes_std": 3.0,
            "source_ip_type": "internal",
            "dest_ip_type": "internal",
            "protocols": ["TCP"],
            "protocol_probs": [1.0],
            "severities": ["info"],
            "severity_probs": [1.0],
        },
    }
    return params[cluster_name]


def _generate_ip(ip_type: str) -> str:
    """
    Generate a synthetic IP address.

    Args:
        ip_type: 'internal' or 'external'

    Returns:
        IP address string
    """
    if ip_type == "internal":
        if np.random.random() < 0.7:
            return f"10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(1, 255)}"
        else:
            return f"192.168.{np.random.randint(0, 256)}.{np.random.randint(1, 255)}"
    else:
        first = np.random.choice([8, 12, 17, 23, 45, 67, 89, 104, 142, 185, 203])
        return f"{first}.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}.{np.random.randint(1, 255)}"


def _generate_port(cluster_name: str, common_ports: Dict) -> int:
    """
    Generate a port number based on cluster type.

    Args:
        cluster_name: Cluster type
        common_ports: Dictionary of port lists

    Returns:
        Port number
    """
    if cluster_name in ["normal_web"]:
        return np.random.choice(common_ports["web"])
    elif cluster_name == "normal_db":
        return np.random.choice(common_ports["database"])
    elif cluster_name == "suspicious_scan":
        if np.random.random() < 0.4:
            return np.random.choice(common_ports["suspicious"])
        else:
            return np.random.randint(1, 65535)
    elif cluster_name == "auth_failure":
        return np.random.choice([22, 389, 636, 3389, 443])
    elif cluster_name == "data_exfil":
        if np.random.random() < 0.5:
            return np.random.choice([21, 22, 443, 8080])
        else:
            return np.random.randint(8000, 9000)
    else:  # maintenance
        return np.random.choice([22, 873, 3306, 5432, 6379])


def _generate_log_message(cluster_name: str) -> str:
    """
    Generate a realistic log message for the cluster type.

    Args:
        cluster_name: Cluster type

    Returns:
        Formatted log message string
    """
    templates = LOG_TEMPLATES[cluster_name]
    template = np.random.choice(templates)

    replacements = {
        "{path}": np.random.choice(["/api/v1/users", "/index.html", "/assets/main.js", "/login"]),
        "{ip}": f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}",
        "{domain}": np.random.choice(["api.example.com", "cdn.service.net", "auth.company.com"]),
        "{asset}": np.random.choice(["logo.png", "styles.css", "app.bundle.js"]),
        "{endpoint}": np.random.choice(["/users", "/orders", "/products", "/auth"]),
        "{ms}": str(np.random.randint(5, 500)),
        "{user_type}": np.random.choice(["admin", "regular", "guest"]),
        "{browser}": np.random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
        "{protocol}": np.random.choice(["HTTP/1.1", "HTTP/2", "HTTPS"]),
        "{table}": np.random.choice(["users", "orders", "products", "sessions"]),
        "{count}": str(np.random.randint(1, 100)),
        "{operation}": np.random.choice(["INSERT", "UPDATE", "DELETE", "SELECT"]),
        "{txn_id}": f"txn_{np.random.randint(10000, 99999)}",
        "{lag}": str(np.random.randint(0, 100)),
        "{db_host}": np.random.choice(["db-primary", "db-replica-1", "db-replica-2"]),
        "{port_count}": str(np.random.randint(10, 1000)),
        "{port}": str(np.random.choice([22, 23, 445, 3389, 80, 443])),
        "{username}": np.random.choice(["admin", "root", "user123", "service_account"]),
        "{size}": str(np.random.randint(1, 1000)),
        "{component}": np.random.choice(["nginx", "postgresql", "redis", "kernel"]),
        "{version}": f"{np.random.randint(1, 5)}.{np.random.randint(0, 20)}.{np.random.randint(0, 10)}",
        "{job_name}": np.random.choice(["backup_db", "cleanup_logs", "sync_data", "health_check"]),
    }

    message = template
    for placeholder, value in replacements.items():
        message = message.replace(placeholder, value)

    return message


if __name__ == "__main__":
    df = generate_network_events(output_path="data/raw/network_events.csv")
    print(f"\nGenerated {len(df):,} events")
    print(f"\nSample:\n{df.head()}")
    print(f"\nCluster distribution:\n{df['true_cluster'].value_counts()}")
