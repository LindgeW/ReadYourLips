# LLMs_api.py

import os
import time  # 1. 导入 time 模块
from typing import Optional, Tuple, Union, List  # 2. 导入 Tuple 用于类型提示
from openai import OpenAI, OpenAIError, Timeout

class VSRConverter:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 timeout: int = 30,
                 provider: str = "volcengine",
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        valid_providers = ["volcengine", "zhipu", "qwen"]
        if provider not in valid_providers:
            raise ValueError(f"不支持的模型提供商: {provider}，可选值: {valid_providers}")
        
        self.provider = provider
        self.timeout = timeout
        
        provider_configs = {
            "volcengine": {
                "env_var": "ARK_API_KEY",
                "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "default_model": "doubao-seed-1-6-flash-250828"
            },
            "zhipu": {
                "env_var": "ZHIPU_API_KEY",
                "default_base_url": "https://open.bigmodel.cn/api/paas/v4/",
                "default_model": "glm-4.5-air"
            },
            "qwen": {
                "env_var": "DASHSCOPE_API_KEY",
                "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "default_model": "qwen-plus-2025-09-11"# qwen-plus-2025-09-11 qwen-flash
            }
        }
        
        config = provider_configs[provider]
        self.api_key = api_key or os.environ.get(config["env_var"])
        if not self.api_key:
            raise ValueError(f"请提供API密钥或设置环境变量{config['env_var']}")
        self.base_url = base_url or config["default_base_url"]
        self.model = model or config["default_model"]
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def convert_vsr_result(self, original_sentence: str, mode: str = 'muti', max_retries: int = 2) -> Tuple[Union[str, List[str]], float]:
        """
        将拼音+声调句子转换为中文句子
        Args:
            original_sentence: 原始唇语识别结果（拼音+声调）
            mode: 'single' (返回一个结果) 或 'multi' (返回最多三个候选项)

        Returns:
            一个元组 (转换结果, API调用耗时秒数)。转换结果可以是字符串或字符串列表。
        """

        single_prompt = """你是一个专业的中文语音识别结果转换引擎，专门处理养老助残领域的语音识别输出。你的任务是将输入的拼音+声调序列转换为准确的中文句子。

**任务背景**：
输入的拼音序列来自唇语识别系统，主要涉及养老、助残、医疗、日常生活等场景。你需要根据拼音准确转换为符合语境的中文句子。

**执行流程**：
1.  **理解语境**：根据养老助残领域的背景理解句子可能的含义
2.  **拼音转汉字**：将拼音序列准确转换为对应的中文句子
3.  **输出**：仅输出转换后的中文句子

**绝对约束**：
- **语境匹配**：转换结果必须符合养老助残领域的语境
- **准确转换**：严格按照拼音转换，确保读音一致
- **纯文本输出**：禁止添加任何解释、标签或编号
- **领域专注**：内容应聚焦于老年人、残疾人的生活、医疗、护理等场景"""

        # -------------------【核心修改区域：重写multi_prompt】-------------------
        multi_prompt = """你是一位顶级的养老助残语音交互专家，专门处理语音识别后的拼音序列（含声调）。用户的表达主要围绕老年人/残障人的日常生活、健康、护理和环境控制。

        【重要背景】
        系统预定义了一组高频标准指令（见下方完整列表），**用户很可能直接说出其中某一句**。但用户也可能：
        - 对指令进行合理组合（如“我 + 肚子痛”）
        - 使用同义表达（如“我想上厕所” ≈ “我要去卫生间”）
        - 说出列表外但语义合理的句子（如“我后背疼”）

        你的任务是：将输入的拼音序列转换为三个**最贴切、最自然、最符合医护场景**的中文短句。

        【完整标准指令集（共160条，ID 0~159）】
        我饿了, 我口渴, 我吃饱了, 水太烫了, 我太累了, 我想睡觉, 我要休息, 扶我起来, 我要上厕所, 我要坐下, 我想吃零食, 我失禁了, 我有点冷, 我好热, 有点闷, 风好大, 打开空调, 关闭空调, 调高温度, 调低温度, 我要吃饭, 我想喝饮料, 我吃不下, 我很好, 我咬不动, 我要起床, 我想吃水果, 我生病了, 紧急呼救, 我该吃药了, 我摔倒了, 我不行了, 我血压高, 我头晕, 我嗓子疼, 我脖子疼, 我腰痛, 我肩膀痛, 我腿疼, 我牙齿疼, 我感冒了, 我发烧了, 打开窗户, 我呼吸困难, 我眼睛难受, 关闭窗户, 把门打开, 把门关上, 把灯打开, 把灯关了, 多久能治好, 需要住院吗, 这药有效吗, 我关节痛, 不用吃药吗, 我心跳太快, 要住院多久, 我一直咳嗽, 我胸闷, 我喘不上气, 情况严重吗, 需要手术吗, 这病传染吗, 我全身乏力, 我要打电话, 我要发短信, 我很开心, 我要聊视频, 谢谢你, 不客气, 我没听清, 是这样的, 我不太清楚, 我很孤独, 没关系, 对不起, 提醒我吃药, 我能行的, 帮我定闹钟, 我要剪头发, 我要洗手, 我要洗澡, 我要锻炼, 我要换衣服, 我要按摩, 我要洗头发, 我想剪指甲, 我看不到, 我要看书, 我要看电视, 我要去运动, 我要玩游戏, 我要下棋, 我要上网, 我想去散步, 我要听音乐, 向前走, 停下来, 向左转, 向右转, 我感到郁闷, 给我家人打电话, 我有点难受, 我肚子痛, 我感到焦虑, 现在几点了, 我不想要那个, 我情况怎么样, 我想去卫生间, 我感觉舒服, 帮我量一下血压, 太亮了, 我很痛, 帮我挪动一下, 太吵了, 医生, 我头痛, 我可以咳嗽吗, 我害怕, 我还好, 我不舒服, 请说慢一点, 我需要住多久, 打扰一下, 我什么时候能回家, 我为什么不能说话, 发生了什么, 我在哪儿, 你们对我做什么, 我身上发生了什么, 和我家人说过了吗, 我想躺下, 我睡不着, 体检什么时候, 怎么用药, 帮我坐起来, 我不能吃这个, 我要喝水, 请再解释一遍, 我受不了了, 快叫医生过来, 我需要一条毯子, 我失眠了, 帮我测下体温, 我有点过敏, 什么时候换药, 疼得更厉害了, 我需要验血吗, 我有点困了, 这个药让我不舒服, 我手脚动不了, 我这里疼, 能扶我一把吗, 我不明白, 能陪我一下吗, 我感到无聊, 我需要打针吗, 帮我换个姿势, 我没胃口

        【生成规则】
        1. **优先匹配标准指令集**：如果拼音与某条指令高度相似，直接输出该指令。
        2. **允许合理泛化**：可生成指令集外但语义合理的句子（如“我后背疼”），但须符合医疗护理场景。
        3. **禁止荒谬输出**：严禁“我生性了”、“感到手了”、“独守”、“可瘦”等不合逻辑的组合。
        4. **三句最好有差异**：若无法生成三个合理变体，允许重复，但不要强行编造。
        5. 若拼音序列为空，直接输出“【无效输入】”或“【请重新输入】”。

        【输出格式】
        严格按：句子一;句子二;句子三  
        用英文分号分隔，无任何额外文字、编号、解释或标点。

        请结合以上完整指令集和医护常识，生成最可能的用户原话。
        """
        # -------------------【核心修改区域结束】-------------------

        system_prompt = multi_prompt if mode == 'multi' else single_prompt
        user_prompt = f"请将以下拼音转换为中文，专注于养老助残和医护领域：{original_sentence}"

        start_time = time.monotonic()

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    timeout=self.timeout
                )
                
                converted_text = response.choices[0].message.content.strip()

                # --- 核心改动：格式验证 ---
                if mode == 'multi':
                    # 检查返回的文本是否包含两个分号，这是正确格式的标志
                    if converted_text and converted_text.count(';') == 2:
                        # 格式正确，跳出重试循环
                        break
                    else:
                        # 格式不正确，记录警告并准备重试
                        print(f"警告: API返回格式不正确 (尝试 {attempt + 1}/{max_retries})。内容: '{converted_text}'。正在重试...")
                        if attempt + 1 == max_retries:
                            print("警告: 已达到最大重试次数，将使用当前不完美的输出。")
                else: # single模式不需要复杂格式，直接跳出
                    break

            except (OpenAIError, Exception) as e:
                end_time = time.monotonic()
                duration = end_time - start_time
                print(f"警告: API调用时发生错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}，耗时 {duration:.2f}s。")
                if attempt + 1 == max_retries:
                    # 如果所有重试都因API错误而失败，则按原样返回
                    if mode == 'multi':
                        return [original_sentence, original_sentence, original_sentence], duration
                    else:
                        return original_sentence, duration
            
            # 短暂休眠，避免立即重试给API造成太大压力
            time.sleep(1)
        
        # --- 重试循环结束，开始处理最终结果 ---
        end_time = time.monotonic()
        duration = end_time - start_time
        
        # 无论重试是否成功，都用最终的 converted_text 进行处理
        if mode == 'multi':
            candidates = converted_text.split(';')
            candidates = [candidate.strip() for candidate in candidates if candidate.strip()]
            while len(candidates) < 3:
                fill_value = candidates[0] if candidates else original_sentence
                candidates.append(fill_value)
            return candidates[:3], duration # 列表输出
            # output_string=""
            # for ans in range(3):  # 输出三个候选项
            #     output_string += candidates[ans] + ';'
            # return output_string,duration
        else:  # single mode
            if not converted_text:
                return original_sentence, duration
            first_sentence = converted_text.split(';')[0].strip()
            return first_sentence if first_sentence else original_sentence, duration



qwen_converter = VSRConverter(
    provider="qwen",
    model="qwen3-max", #  qwen-plus-2025-09-11 qwen3-max
    # 请在此处填入你的有效API Key
    api_key=os.environ.get("DASHSCOPE_API_KEY", "sk-26e71b53faf94708b012fae038668845")
)


'''
if __name__ == "__main__":
    try:
        # 推荐使用性能更强的模型来执行复杂的指令，比如 qwen-plus
        # 如果使用 qwen-flash，效果可能会打折扣
        print("\n===== 百炼(Qwen) =====")
        qwen_converter = VSRConverter(
            provider="qwen",
            model="qwen3-max", #  qwen-plus-2025-09-11 qwen3-max
            # 请在此处填入你的有效API Key
            api_key=os.environ.get("DASHSCOPE_API_KEY", "sk-26e71b53faf94708b012fae038668845")
        )

        test_pinyin = "wo3 men5 qu4 gong1 yuan2 wan2 ba5",   # 我们去公园玩吧

        converted, duration = qwen_converter.convert_vsr_result(test_pinyin, mode="multi")
        print(f"拼音: {test_pinyin}")
        print(f"转换: {converted} (耗时: {duration:.2f}s)")
        print("-" * 50)
    except Exception as e:
        print(f"程序运行失败: {str(e)}")
'''
