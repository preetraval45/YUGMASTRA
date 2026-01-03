#!/bin/bash

# Database Restore Script for YUGMASTRA
# Restores PostgreSQL database from backup

set -e

# Configuration
BACKUP_DIR="/backups/postgres"
CONTAINER_NAME="yugmastra-postgres-1"
DATABASE_NAME="yugmastra"

# Check if backup file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Available backups:"
    ls -lh "$BACKUP_DIR"/yugmastra_backup_*.sql.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"

# Check if backup file exists
if [ ! -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_DIR/$BACKUP_FILE"
    exit 1
fi

echo "[$(date)] Starting database restore from: $BACKUP_FILE"

# Confirmation prompt
read -p "WARNING: This will OVERWRITE the current database. Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

# Drop and recreate database
echo "[$(date)] Dropping existing database..."
docker exec -t "$CONTAINER_NAME" psql -U yugmastra -c "DROP DATABASE IF EXISTS ${DATABASE_NAME};"
docker exec -t "$CONTAINER_NAME" psql -U yugmastra -c "CREATE DATABASE ${DATABASE_NAME};"

# Restore from backup
echo "[$(date)] Restoring database..."
gunzip -c "$BACKUP_DIR/$BACKUP_FILE" | docker exec -i "$CONTAINER_NAME" psql -U yugmastra -d "$DATABASE_NAME"

if [ $? -eq 0 ]; then
    echo "[$(date)] Database restored successfully from: $BACKUP_FILE"
else
    echo "[$(date)] ERROR: Restore failed!"
    exit 1
fi
