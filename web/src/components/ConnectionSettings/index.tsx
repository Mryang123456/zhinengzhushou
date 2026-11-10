import { useState } from "react";
import { Alert, Button, Form, Input, Modal } from "antd";
import { connectionLabels, type ConnectionStatus } from "../../services/agentApi";
import "./style.css";

type Props = {
  apiUrl: string;
  apiKey: string;
  userId: string;
  connectionStatus: ConnectionStatus;
  checking: boolean;
  onCheckConnection: () => void;
  onApiKey: (value: string) => void;
  onLogin: (username: string, password: string) => Promise<void>;
  onClose: () => void;
};
type Credentials = { username: string; password: string };

export function ConnectionSettings({ apiUrl, apiKey, userId, connectionStatus, checking, onCheckConnection, onApiKey, onLogin, onClose }: Props) {
  const [form] = Form.useForm<Credentials>();
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async ({ username, password }: Credentials) => {
    setError("");
    setSubmitting(true);
    try {
      await onLogin(username.trim(), password);
      form.resetFields(["password"]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "登录失败，请重试。");
    } finally {
      setSubmitting(false);
    }
  };

  return <Modal open title="连接与登录" footer={null} onCancel={onClose} width={440}>
    <div className="connection-settings-check">
      <span role="status">{connectionLabels[checking ? "checking" : connectionStatus]}</span>
      <Button size="small" loading={checking} onClick={onCheckConnection}>重新检测</Button>
    </div>
    <Form form={form} layout="vertical" onFinish={submit}>
      <Form.Item label="服务地址" htmlFor="agent-api-url">
        <Input id="agent-api-url" value={apiUrl} readOnly />
      </Form.Item>
      <Form.Item label="访问密钥" htmlFor="agent-api-key">
        <Input.Password id="agent-api-key" visibilityToggle={false} value={apiKey} onChange={(e) => onApiKey(e.target.value)} placeholder="可选的服务访问密钥" />
      </Form.Item>
      <Form.Item label="用户名" name="username" rules={[{ required: true, whitespace: true, message: "请输入用户名" }]}>
        <Input placeholder={`当前：${userId}`} autoComplete="username" />
      </Form.Item>
      <Form.Item label="密码" name="password" rules={[{ required: true, message: "请输入密码" }]}>
        <Input.Password autoComplete="current-password" />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit" loading={submitting}>获取登录令牌</Button>
      </Form.Item>
      {error ? <Alert type="error" title={error} showIcon /> : <p>访问令牌仅保存在当前页面内存中，刷新页面后自动清除。</p>}
    </Form>
  </Modal>;
}
