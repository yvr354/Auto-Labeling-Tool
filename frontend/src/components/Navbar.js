import React from 'react';
import { Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  RobotOutlined,
  ProjectOutlined,
  DatabaseOutlined,
  EditOutlined,
  BulbOutlined
} from '@ant-design/icons';

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/models',
      icon: <RobotOutlined />,
      label: 'Models',
    },
    {
      key: '/projects',
      icon: <ProjectOutlined />,
      label: 'Projects',
    },
    {
      key: '/datasets',
      icon: <DatabaseOutlined />,
      label: 'Datasets',
    },
    {
      key: '/annotate',
      icon: <EditOutlined />,
      label: 'Annotate',
    },
    {
      key: '/active-learning',
      icon: <BulbOutlined />,
      label: 'Active Learning',
    },
  ];

  const handleMenuClick = ({ key }) => {
    navigate(key);
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
      <div style={{ 
        color: 'white', 
        fontSize: '20px', 
        fontWeight: 'bold', 
        marginRight: '40px',
        marginLeft: '24px'
      }}>
        🏷️ Auto-Labeling-Tool
      </div>
      <Menu
        theme="dark"
        mode="horizontal"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        style={{ flex: 1, minWidth: 0 }}
      />
    </div>
  );
};

export default Navbar;