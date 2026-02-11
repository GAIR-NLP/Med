#!/usr/bin/env bash

#=======================================================================
#
#   This script retrieves the IP address of the Ray Head Node and
#   displays it in a formatted box.
#
#=======================================================================

# --- Configuration ---
LABEL="✨ Head Node IP:"
ERROR_TITLE="❌ Error"
ERROR_MSG_L1="Could not retrieve the Head Node IP address."
ERROR_MSG_L2="Please check if the Ray cluster is running."
INFO_ICON="💡"

# --- Main Logic ---
echo "Fetching Ray Head Node IP, please wait..."

# Execute the command and capture the IP address.
# Redirect stderr to /dev/null to hide potential errors from the command itself.
IP_ADDRESS=$(ray list nodes --filter is_head_node=True | awk 'NR==10 {print $3}' 2>/dev/null)

# --- Error Handling ---
# Check if the IP_ADDRESS variable is empty
if [[ -z "$IP_ADDRESS" ]]; then
    # Determine box width based on the longest error message line
    if [[ ${#ERROR_MSG_L1} -gt ${#ERROR_MSG_L2} ]]; then
      width=$((${#ERROR_MSG_L1} + 4))
    else
      width=$((${#ERROR_MSG_L2} + 4))
    fi

    # Draw the error box
    echo -e "\n╭─${ERROR_TITLE} $(printf '─%.0s' $(seq 1 $(($width - ${#ERROR_TITLE} - 3))))╮"
    echo "│ $(printf '%*s' $width) │"
    # Pad each line to fit the calculated width
    printf "│  %-*s  │\n" "$((width-4))" "$ERROR_MSG_L1"
    printf "│  %-*s  │\n" "$((width-4))" "$ERROR_MSG_L2"
    echo "│ $(printf '%*s' $width) │"
    echo -e "╰$(printf '─%.0s' $(seq 1 $(($width + 2))))╯\n"
    exit 1
fi

# --- Success Display ---
# Dynamically calculate the content and box width based on the label and IP address length
content_width=$((${#LABEL} + ${#IP_ADDRESS} + 2)) # Label + Space + IP
box_width=$(($content_width + 4)) # Add padding for left/right margins

# Draw the display box
echo -e "\n╭$(printf '─%.0s' $(seq 1 $box_width))╮"
echo "│$(printf '%*s' $box_width)│"
echo "│  ${LABEL}  ${IP_ADDRESS}  │"
echo "│$(printf '%*s' $box_width)│"
echo -e "╰$(printf '─%.0s' $(seq 1 $box_width))╯\n"


# --- Final Instruction ---
echo "${INFO_ICON} Please copy the IP address from the box above."
echo "" # Print a final newline for a clean exit
