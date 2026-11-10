import { useRef } from "react";
import type { ComponentRef } from "react";
import { Sender } from "@ant-design/x";
import { Button } from "antd";
import { Database, Paperclip, Send, Wrench, X } from "lucide-react";

type Props = {
  value: string;
  isSending: boolean;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  onStop: () => void;
  onTool: () => void;
  onMemory: () => void;
};

export function Composer({ value, isSending, onChange, onSubmit, onStop, onTool, onMemory }: Props) {
  const sender = useRef<ComponentRef<typeof Sender>>(null);
  const activate = (action: () => void) => {
    action();
    sender.current?.focus();
  };

  return <div className="chat-composer">
    <Sender
      ref={sender}
      value={value}
      readOnly={isSending}
      loading={isSending}
      onCancel={onStop}
      onChange={onChange}
      onSubmit={onSubmit}
      autoSize={{ minRows: 2, maxRows: 6 }}
      placeholder="输入你的问题，支持查询、规划、工具调用与记忆…"
      onKeyDown={(event) => event.nativeEvent.isComposing ? false : undefined}
      suffix={(_, { components: { SendButton, LoadingButton } }) => isSending
        ? <LoadingButton danger shape="default" icon={<X size={17}/>}>停止</LoadingButton>
        : <SendButton shape="default" icon={<Send size={17}/>} disabled={!value.trim()}>发送</SendButton>}
      footer={<div className="chat-composer-tools">
        <Button type="text" size="small" icon={<Wrench size={16}/>} disabled={isSending} onClick={() => activate(onTool)}>工具按需调用</Button>
        <Button type="text" size="small" icon={<Database size={16}/>} disabled={isSending} onClick={() => activate(onMemory)}>长期记忆</Button>
        <Button type="text" size="small" icon={<Paperclip size={16}/>} disabled aria-label="添加附件（暂未启用）" title="附件上传即将支持"/>
      </div>}
    />
  </div>;
}
