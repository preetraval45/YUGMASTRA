#!/bin/bash

# Database Backup Script for YUGMASTRA
# Backs up PostgreSQL database with compression and retention policy

set -e

# Configuration
BACKUP_DIR="/backups/postgres"
CONTAINER_NAME="yugmastra-postgres-1"
DATABASE_NAME="yugmastra"
RETENTION_DAYS=7
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="yugmastra_backup_${TIMESTAMP}.sql.gz"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting database backup..."

# Perform backup using pg_dump inside container
docker exec -t "$CONTAINER_NAME" pg_dump -U yugmastra -d "$DATABASE_NAME" | gzip > "$BACKUP_DIR/$BACKUP_FILE"

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo "[$(date)] Backup successful: $BACKUP_FILE"

    # Calculate backup size
    BACKUP_SIZE=$(du -h "$BACKUP_DIR/$BACKUP_FILE" | cut -f1)
    echo "[$(date)] Backup size: $BACKUP_SIZE"

    # Remove backups older than retention period
    echo "[$(date)] Removing backups older than $RETENTION_DAYS days..."
    find "$BACKUP_DIR" -name "yugmastra_backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete

    # List remaining backups
    echo "[$(date)] Current backups:"
    ls -lh "$BACKUP_DIR"/yugmastra_backup_*.sql.gz 2>/dev/null || echo "No backups found"

    echo "[$(date)] Backup process completed successfully"
else
    echo "[$(date)] ERROR: Backup failed!"
    exit 1
fi
