import glob
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Trajectory Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_trajectory_data(trajectory_dir):
    """Load all trajectory data from JSONL files"""
    trajectory_data = {}

    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(trajectory_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        category = os.path.basename(jsonl_file).replace("_trajectories.jsonl", "")
        trajectory_data[category] = []

        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trajectory_data[category].append(json.loads(line))

    return trajectory_data


def display_trajectory_card(trajectory, index):
    """Display a single trajectory card"""
    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### Trajectory #{index + 1}")
            st.markdown(f"**UID:** `{trajectory['uid']}`")

            # Metadata information
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("Response Length", trajectory["response_length"])
            with metrics_col2:
                st.metric("Tool Calls", trajectory["tool_call_counts"])
            with metrics_col3:
                accuracy = trajectory.get("accuracy_reward", "N/A")
                st.metric(
                    "Accuracy",
                    f"{accuracy:.2f}" if isinstance(accuracy, (int, float)) else accuracy,
                )
            with metrics_col4:
                total_reward = trajectory.get("total_reward", "N/A")
                st.metric(
                    "Total Reward",
                    f"{total_reward:.2f}"
                    if isinstance(total_reward, (int, float))
                    else total_reward,
                )

        with col2:
            # Display related image thumbnails
            images_info = []
            for msg in trajectory.get("messages", []):
                for content in msg.get("content", []):
                    if content.get("type") == "image" and "image_path" in content:
                        images_info.append(content["image_path"])

            if images_info:
                st.markdown(f"**Related Images:** {len(images_info)} images")
                if st.button(f"View Images #{index + 1}", key=f"view_images_{index}"):
                    st.session_state[f"show_images_{index}"] = True


def render_masked_text(text):
    """Convert <MASK_START>...<MASK_END> to styled spans, handling line breaks properly"""
    import re

    def replace_masked_block(match):
        content = match.group(1)
        lines = content.split("\n")
        styled_lines = []
        for line in lines:
            if line.strip():  # 非空行
                styled_lines.append(
                    f'<span style="text-decoration: line-through; background-color: #ffe6e6; color: #666; padding: 2px 4px; border-radius: 3px; font-weight: bold; opacity: 0.8;">{line}</span>'
                )
            else:  # 空行
                styled_lines.append("<br>")
        return "<br>".join(styled_lines)

    return re.sub(r"<MASK_START>(.*?)<MASK_END>", replace_masked_block, text, flags=re.DOTALL)


def display_trajectory_details(trajectory, index, base_dir):
    """Display detailed trajectory information"""
    st.markdown("---")

    # Text display tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Marked Text", "Full Text", "Messages", "Details"])

    with tab1:
        st.markdown("### Text with Response Mask Markers")
        rendered_text = render_masked_text(trajectory["marked_text"])
        st.markdown(rendered_text, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Full Valid Text")
        st.text_area("", trajectory["full_valid_text"], height=300, key=f"full_text_{index}")

    with tab3:
        st.markdown("### Messages History")
        for i, msg in enumerate(trajectory.get("messages", [])):
            with st.expander(f"Message {i + 1} - {msg['role']}"):
                for content in msg.get("content", []):
                    if content["type"] == "text":
                        st.markdown(content["text"])
                    elif content["type"] == "image":
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            # Try to display image
                            image_path = os.path.join(base_dir, content["image_path"])
                            if os.path.exists(image_path):
                                try:
                                    img = Image.open(image_path)
                                    st.image(img, width=200, caption=content.get("description", ""))
                                except Exception as e:
                                    st.error(f"Cannot load image: {e}")
                            else:
                                st.warning(f"Image file not found: {content['image_path']}")
                        with col2:
                            st.markdown(f"**Description:** {content.get('description', 'N/A')}")

    with tab4:
        st.markdown("### Detailed Information")

        # Position IDs
        st.markdown("**Position IDs (compressed):**")
        st.json(trajectory.get("position_ids", []))

        # Result Dict
        if "result_dict" in trajectory:
            st.markdown("**Result Dictionary:**")
            st.json(trajectory["result_dict"])

        # All reward-related fields
        reward_fields = {k: v for k, v in trajectory.items() if "reward" in k.lower()}
        if reward_fields:
            st.markdown("**Reward-related Fields:**")
            st.json(reward_fields)


def display_statistics(trajectory_data):
    """Display statistical overview"""
    st.markdown("## 📊 Statistical Overview")

    # Aggregate all data
    all_trajectories = []
    for category, trajectories in trajectory_data.items():
        for traj in trajectories:
            traj_copy = traj.copy()
            traj_copy["category"] = category
            all_trajectories.append(traj_copy)

    if not all_trajectories:
        st.warning("No trajectory data found")
        return

    df = pd.DataFrame(all_trajectories)

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trajectories", len(all_trajectories))
    with col2:
        avg_length = df["response_length"].mean()
        st.metric("Avg Response Length", f"{avg_length:.1f}")
    with col3:
        total_tool_calls = df["tool_call_counts"].sum()
        st.metric("Total Tool Calls", total_tool_calls)
    with col4:
        if "accuracy_reward" in df.columns:
            accuracy_rate = df["accuracy_reward"].mean()
            st.metric("Avg Accuracy", f"{accuracy_rate:.2f}")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Response length distribution
        fig_length = px.histogram(
            df,
            x="response_length",
            color="category",
            title="Response Length Distribution",
            labels={"response_length": "Response Length", "count": "Count"},
        )
        st.plotly_chart(fig_length, use_container_width=True)

    with chart_col2:
        # Tool call count distribution
        fig_tools = px.histogram(
            df,
            x="tool_call_counts",
            color="category",
            title="Tool Call Count Distribution",
            labels={"tool_call_counts": "Tool Call Count", "count": "Count"},
        )
        st.plotly_chart(fig_tools, use_container_width=True)

    # Category statistics
    st.markdown("### Statistics by Category")
    category_stats = (
        df.groupby("category")
        .agg(
            {
                "response_length": ["count", "mean", "max"],
                "tool_call_counts": ["sum", "mean"],
                "accuracy_reward": "mean" if "accuracy_reward" in df.columns else lambda x: None,
            }
        )
        .round(2)
    )

    st.dataframe(category_stats, use_container_width=True)


def get_exp_names(trajectories_base_dir="./trajectories"):
    """Get all experiment names from trajectories directory"""
    if not os.path.exists(trajectories_base_dir):
        return []

    exp_names = []
    for item in os.listdir(trajectories_base_dir):
        item_path = os.path.join(trajectories_base_dir, item)
        if os.path.isdir(item_path):
            exp_names.append(item)
    return sorted(exp_names)


def get_exp_steps(trajectories_base_dir, exp_name):
    """Get all step directories for a given experiment"""
    exp_path = os.path.join(trajectories_base_dir, exp_name)
    if not os.path.exists(exp_path):
        return []

    steps = []
    for item in os.listdir(exp_path):
        item_path = os.path.join(exp_path, item)
        if os.path.isdir(item_path) and item.startswith(exp_name):
            steps.append(item)
    return sorted(steps)


def main():
    st.title("🎯 Trajectory Visualizer")
    st.markdown("---")

    # Sidebar - Select data directory
    st.sidebar.title("🔧 Configuration")

    # Dropdown for experiment selection
    trajectories_base_dir = "./trajectories"
    exp_names = get_exp_names(trajectories_base_dir)

    if not exp_names:
        st.error(f"No experiment directories found in {trajectories_base_dir}")
        return

    selected_exp = st.sidebar.selectbox(
        "Select Experiment", exp_names, help="Choose the experiment directory"
    )

    # Dropdown for step selection
    exp_steps = get_exp_steps(trajectories_base_dir, selected_exp)

    if not exp_steps:
        st.error(f"No step directories found for experiment {selected_exp}")
        return

    selected_step = st.sidebar.selectbox(
        "Select Experiment Step", exp_steps, help="Choose the experiment step directory"
    )

    # Construct trajectory directory path
    trajectory_dir = os.path.join(trajectories_base_dir, selected_exp, selected_step)

    # Display current selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Selection:**")
    st.sidebar.markdown(f"📁 Experiment: `{selected_exp}`")
    st.sidebar.markdown(f"📂 Step: `{selected_step}`")
    st.sidebar.markdown(f"📄 Path: `{trajectory_dir}`")

    if not os.path.exists(trajectory_dir):
        st.error(f"Directory does not exist: {trajectory_dir}")
        return

    # Load data
    try:
        trajectory_data = load_trajectory_data(trajectory_dir)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if not trajectory_data:
        st.warning("No trajectory data files found")
        return

    # Sidebar - Select view mode
    view_mode = st.sidebar.radio("View Mode", ["Statistical Overview", "Trajectory Browser"])

    if view_mode == "Statistical Overview":
        display_statistics(trajectory_data)

    else:  # Trajectory Browser
        # Sidebar - Select category
        categories = list(trajectory_data.keys())
        selected_category = st.sidebar.selectbox(
            "Select Trajectory Category", categories, help="Choose the trajectory category to view"
        )

        if selected_category not in trajectory_data:
            st.error(f"Category not found: {selected_category}")
            return

        trajectories = trajectory_data[selected_category]

        if not trajectories:
            st.warning(f"No trajectory data in category '{selected_category}'")
            return

        # Main interface
        st.markdown(f"## 📁 {selected_category}")
        st.markdown(f"Total **{len(trajectories)}** trajectories")

        # Display trajectory list
        for i, trajectory in enumerate(trajectories):
            with st.expander(f"Trajectory #{i + 1} - {trajectory['uid'][:8]}...", expanded=False):
                display_trajectory_card(trajectory, i)

                # Details button
                if st.button(f"View Details #{i + 1}", key=f"details_{i}"):
                    st.session_state[f"show_details_{i}"] = True

                # Show details
                if st.session_state.get(f"show_details_{i}", False):
                    display_trajectory_details(trajectory, i, trajectory_dir)
                    if st.button(f"Hide Details #{i + 1}", key=f"hide_details_{i}"):
                        st.session_state[f"show_details_{i}"] = False


if __name__ == "__main__":
    main()
