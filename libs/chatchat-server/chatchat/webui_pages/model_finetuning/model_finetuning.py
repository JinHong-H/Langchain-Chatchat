import streamlit as st
import time
import pandas as pd
from chatchat.webui_pages.utils import *


def model_finetuning_page(api: ApiRequest):
    st.title("🤖 P-Tuning v2 模型微调")
    
    # 初始化session state
    if "finetuning_status" not in st.session_state:
        st.session_state.finetuning_status = "idle"  # idle, running, completed, error
    
    if "finetuning_progress" not in st.session_state:
        st.session_state.finetuning_progress = 0
    
    if "finetuning_log" not in st.session_state:
        st.session_state.finetuning_log = []
    
    if "finetuning_job_id" not in st.session_state:
        st.session_state.finetuning_job_id = None
    
    # 侧边栏配置 - 专门针对P-Tuning v2
    with st.sidebar:
        st.title("⚙️ P-Tuning v2 配置")
        
        # 模型选择
        model_options = ["THUDM/chatglm3-6b", "baichuan-inc/Baichuan2-7B-Chat", "Qwen/Qwen-7B-Chat", "internlm/internlm-chat-7b"]
        selected_model = st.selectbox("基础模型", model_options, index=0)
        
        # 数据集上传
        st.subheader("训练数据集")
        dataset_file = st.file_uploader("上传数据集", type=["json", "jsonl", "csv"], 
                                       help="支持JSON、JSONL、CSV格式的数据集文件")
        
        if dataset_file is not None:
            st.success(f"已上传: {dataset_file.name}")
        
        # P-Tuning v2 专属参数
        st.subheader("P-Tuning v2 参数")
        pre_seq_len = st.slider("前缀长度", 8, 128, 16, 
                               help="前缀序列长度，控制可学习参数数量")
        prefix_projection = st.checkbox("启用前缀投影", value=True,
                                       help="是否使用前缀投影层")
        encoder_hidden_size = st.selectbox("编码器隐藏层大小", [64, 128, 256, 512], index=1,
                                          help="前缀编码器的隐藏层维度")
        
        # 训练参数
        st.subheader("训练参数")
        epochs = st.number_input("训练轮数", min_value=1, max_value=100, value=3)
        batch_size = st.selectbox("批次大小", [1, 2, 4, 8, 16], index=2)
        learning_rate = st.select_slider("学习率", 
                                        options=["1e-5", "2e-5", "5e-5", "1e-4", "2e-4", "5e-4"],
                                        value="2e-5")
        warmup_ratio = st.slider("Warmup 比例", 0.0, 0.5, 0.1, 0.05,
                                help="学习率预热比例")
        
        # 保存配置
        output_dir = st.text_input("模型保存路径", "./models/finetuned_model")
        
        start_finetuning = st.button("🚀 开始 P-Tuning v2 微调", 
                                    disabled=st.session_state.finetuning_status == "running" or dataset_file is None,
                                    type="primary",
                                    use_container_width=True)
        
        if st.session_state.finetuning_status == "running":
            st.warning("微调正在进行中，请等待完成...")
            st.progress(st.session_state.finetuning_progress)
            
            # 轮询训练状态
            poll_training_status(api)
            
            # 添加停止训练按钮
            if st.button("⏹️ 停止训练", use_container_width=True):
                if st.session_state.finetuning_job_id:
                    # 调用API停止训练
                    try:
                        response = api.post("/finetuning/stop", data={
                            "job_id": st.session_state.finetuning_job_id
                        })
                        if response.get("code") == 200:
                            st.session_state.finetuning_status = "idle"
                            st.session_state.finetuning_job_id = None
                            st.success("训练已停止")
                            st.rerun()
                        else:
                            st.error("停止训练失败")
                    except Exception as e:
                        st.error(f"停止训练时出错: {e}")
        
        if st.session_state.finetuning_status == "completed":
            st.success("✅ P-Tuning v2 微调已完成!")
            if st.button("🔄 重新开始", use_container_width=True):
                st.session_state.finetuning_status = "idle"
                st.session_state.finetuning_progress = 0
                st.session_state.finetuning_log = []
                st.session_state.finetuning_job_id = None
                st.rerun()
    
    # 主内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["训练配置", "P-Tuning 参数", "训练日志", "模型评估"])
    
    with tab1:
        st.subheader("训练配置详情")
        config_data = {
            "配置项": ["基础模型", "训练轮数", "批次大小", "学习率", "Warmup比例", "数据集"],
            "值": [selected_model, str(epochs), str(batch_size), learning_rate, 
                  str(warmup_ratio), dataset_file.name if dataset_file else "未选择"]
        }
        config_df = pd.DataFrame(config_data)
        st.table(config_df)
        
        # 添加P-Tuning v2说明
        st.subheader("💡 P-Tuning v2 介绍")
        st.markdown("""
        P-Tuning v2 是一种高效的参数微调方法，具有以下特点：
        - **参数高效**: 仅优化前缀参数，冻结预训练模型的其他参数
        - **资源友好**: 相比全量微调，显著减少显存占用和训练时间
        - **效果优秀**: 在多种下游任务上表现接近全量微调
        
        **适用场景**:
        - 显存资源有限
        - 快速适应新任务
        - 多任务模型部署
        """)
    
    with tab2:
        st.subheader("P-Tuning v2 参数详情")
        ptuning_config_data = {
            "参数": ["前缀长度", "启用前缀投影", "编码器隐藏层大小"],
            "值": [str(pre_seq_len), "是" if prefix_projection else "否", str(encoder_hidden_size)],
            "说明": [
                "控制可学习前缀参数的数量",
                "是否使用前缀投影层增加表达能力",
                "前缀编码器的隐藏层维度"
            ]
        }
        ptuning_config_df = pd.DataFrame(ptuning_config_data)
        st.table(ptuning_config_df)
        
        # 参数调优建议
        st.subheader("🔧 参数调优建议")
        st.markdown("""
        - **前缀长度**: 较大值(64-128)适用于复杂任务，较小值(8-32)适用于简单任务
        - **前缀投影**: 建议启用，可提升模型表达能力
        - **编码器大小**: 通常设置为128或256，过大会增加参数量
        """)
    
    with tab3:
        st.subheader("训练日志")
        
        # 显示实时日志
        log_container = st.container()
        with log_container:
            for log_entry in st.session_state.finetuning_log:
                st.text(log_entry)
        
        # 如果正在训练，显示状态信息
        if st.session_state.finetuning_status == "running":
            st.info("正在训练中...")
    
    with tab4:
        st.subheader("模型评估")
        st.info("训练完成后将在此展示模型评估结果")
        
        if st.session_state.finetuning_status == "completed":
            # 示例评估结果
            st.metric("最终损失", "0.45", delta="-0.55")
            st.metric("验证准确率", "92.3%", delta="15.2%")
            
            # 示例对比表格
            comparison_data = {
                "评估指标": ["准确率", "F1分数", "BLEU分数"],
                "原始模型": ["77.1%", "72.5%", "12.3"],
                "微调模型": ["92.3%", "89.7%", "32.1"]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
            
            st.subheader("💾 模型使用")
            st.markdown(f"""
            微调完成的模型已保存至: `{output_dir}`
            
            您可以在以下场景使用该模型:
            1. 在对话页面选择该模型进行推理
            2. 部署为独立服务
            3. 进一步微调或优化
            """)

    # 处理开始微调按钮点击事件
    if start_finetuning:
        st.session_state.finetuning_status = "running"
        st.session_state.finetuning_progress = 0
        st.session_state.finetuning_log = ["开始准备 P-Tuning v2 训练任务..."]
        
        # 准备训练参数
        finetuning_params = {
            "model_name": selected_model,
            "method": "P-Tuning v2",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "output_dir": output_dir,
            "pre_seq_len": pre_seq_len,
            "prefix_projection": prefix_projection,
            "encoder_hidden_size": encoder_hidden_size
        }
        
        # 如果有上传的数据集文件，需要先上传
        if dataset_file is not None:
            try:
                # 上传数据集文件
                files = {"file": (dataset_file.name, dataset_file.getvalue(), dataset_file.type)}
                response = api.post("/finetuning/upload_dataset", files=files)
                
                if response.get("code") == 200:
                    dataset_info = response
                    finetuning_params["dataset_path"] = dataset_info.get("path")
                else:
                    st.error("数据集上传失败")
                    st.session_state.finetuning_status = "idle"
                    return
            except Exception as e:
                st.error(f"上传数据集时出错: {e}")
                st.session_state.finetuning_status = "idle"
                return
        
        # 启动微调任务
        try:
            response = api.post("/finetuning/start", json=finetuning_params)
            
            if response.get("code") == 200:
                result = response
                st.session_state.finetuning_job_id = result.get("job_id")
                st.success("P-Tuning v2 微调任务已启动")
            else:
                st.error("启动微调任务失败")
                st.session_state.finetuning_status = "idle"
        except Exception as e:
            st.error(f"启动微调任务时出错: {e}")
            st.session_state.finetuning_status = "idle"
        
        st.rerun()


def poll_training_status(api: ApiRequest):
    """轮询训练状态"""
    if st.session_state.finetuning_job_id:
        try:
            response = api.get(f"/finetuning/status/{st.session_state.finetuning_job_id}")
            if response.get("code") == 200:
                data = response.get("data", {})
                status = data.get("status", "running")
                
                # 更新日志
                logs = data.get("logs", [])
                if logs:
                    st.session_state.finetuning_log = logs
                
                # 更新进度 (简化处理)
                if status == "completed":
                    st.session_state.finetuning_status = "completed"
                    st.session_state.finetuning_progress = 1.0
                    st.rerun()
                elif status == "error":
                    st.session_state.finetuning_status = "error"
                    st.rerun()
                elif status == "stopped":
                    st.session_state.finetuning_status = "idle"
                    st.session_state.finetuning_job_id = None
                    st.rerun()
        except Exception as e:
            st.warning(f"获取训练状态时出错: {e}")