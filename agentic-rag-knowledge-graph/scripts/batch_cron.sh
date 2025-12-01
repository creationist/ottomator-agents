#!/bin/bash
#
# Batch content generation cron script
# 
# This script triggers batch content generation via the API.
# Set up cron jobs to call this script at appropriate intervals.
#
# Cron schedule recommendations:
#   Monthly General:    0 0 1 * *     (1st of month at midnight)
#   Monthly Personal:   0 1 1 * *     (1st of month at 1 AM, after general)
#   Moon Reflections:   0 */12 * * *  (Every 12 hours for Moon sign changes)
#
# Example crontab entries:
#   0 0 1 * * /path/to/scripts/batch_cron.sh monthly
#   0 */12 * * * /path/to/scripts/batch_cron.sh moon
#

set -e

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost:8058}"
LOG_FILE="${LOG_FILE:-/var/log/astro-content-batch.log}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if API is available
check_api() {
    if ! curl -s --fail "${API_BASE_URL}/health" > /dev/null; then
        log "ERROR: API not available at ${API_BASE_URL}"
        exit 1
    fi
}

# Generate monthly content (general + personal for all users)
generate_monthly() {
    log "Starting monthly content generation..."
    
    # Get current year and month if not specified
    YEAR=${1:-$(date +%Y)}
    MONTH=${2:-$(date +%m)}
    
    log "Generating for ${YEAR}-${MONTH}"
    
    response=$(curl -s -X POST "${API_BASE_URL}/content/batch/monthly?year=${YEAR}&month=${MONTH}" \
        -H "Content-Type: application/json")
    
    log "Monthly batch response: $response"
    
    # Check for success
    if echo "$response" | grep -q '"status":"completed"'; then
        log "Monthly content generation completed successfully"
    else
        log "WARNING: Monthly content generation may have issues"
    fi
}

# Generate moon reflection questions
generate_moon_reflection() {
    log "Starting moon reflection generation..."
    
    response=$(curl -s -X POST "${API_BASE_URL}/content/batch/generate" \
        -H "Content-Type: application/json" \
        -d '{"content_type": "moon_reflection"}')
    
    log "Moon reflection batch response: $response"
    
    if echo "$response" | grep -q '"status":"completed"'; then
        log "Moon reflection generation completed successfully"
    else
        log "WARNING: Moon reflection generation may have issues"
    fi
}

# Generate specific content type
generate_content() {
    local content_type=$1
    log "Starting ${content_type} generation..."
    
    response=$(curl -s -X POST "${API_BASE_URL}/content/batch/generate" \
        -H "Content-Type: application/json" \
        -d "{\"content_type\": \"${content_type}\"}")
    
    log "${content_type} batch response: $response"
}

# Main
main() {
    log "=== Batch Content Generation Started ==="
    log "API URL: ${API_BASE_URL}"
    
    check_api
    
    case "${1:-help}" in
        monthly)
            generate_monthly "$2" "$3"
            ;;
        moon)
            generate_moon_reflection
            ;;
        general)
            generate_content "monthly_general"
            ;;
        personal)
            generate_content "monthly_personal"
            ;;
        all)
            generate_monthly
            generate_moon_reflection
            ;;
        help|*)
            echo "Usage: $0 {monthly|moon|general|personal|all} [year] [month]"
            echo ""
            echo "Commands:"
            echo "  monthly [year] [month]  - Generate all monthly content"
            echo "  moon                    - Generate moon reflection questions"
            echo "  general                 - Generate general monthly content only"
            echo "  personal                - Generate personal monthly content only"
            echo "  all                     - Generate all content types"
            echo ""
            echo "Environment variables:"
            echo "  API_BASE_URL  - API base URL (default: http://localhost:8058)"
            echo "  LOG_FILE      - Log file path (default: /var/log/astro-content-batch.log)"
            exit 0
            ;;
    esac
    
    log "=== Batch Content Generation Finished ==="
}

main "$@"

