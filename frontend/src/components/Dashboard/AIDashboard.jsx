import React, { Suspense, useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Environment, Float, Stars } from '@react-three/drei';
import { motion } from 'framer-motion';
import { 
    Box, 
    Typography, 
    Paper, 
    Grid, 
    Container, 
    Stack 
} from '@mui/material';
import * as THREE from 'three';

function StarBackground() {
    const { scene } = useThree();
    
    useEffect(() => {
        scene.background = new THREE.Color(0x0a0a2a);
    }, [scene]);

    return (
        <>
            <Stars 
                radius={300}
                depth={100}
                count={5000}
                factor={4}
                saturation={0}
                fade
                speed={0.2}
            />
            <ambientLight intensity={0.2} />
            <directionalLight 
                position={[10, 10, 5]} 
                intensity={1} 
                color="#4a6cf7"
            />
        </>
    );
}

function AIAssistant() {
    const groupRef = useRef();

    useFrame((state) => {
        const time = state.clock.getElapsedTime();
        if (groupRef.current) {
            groupRef.current.rotation.y = time * 0.3;
        }
    });

    return (
        <Float
            speed={1.5}
            rotationIntensity={0.4}
            floatIntensity={0.4}
            floatingRange={[0, 0.3]}
        >
            <group ref={groupRef}>
                {/* Body */}
                <mesh position={[0, 0, 0]} castShadow>
                    <sphereGeometry args={[1, 32, 32]} />
                    <meshStandardMaterial 
                        color="#3a5fff" 
                        metalness={0.6} 
                        roughness={0.4} 
                        emissive="#2a4aff"
                        emissiveIntensity={0.3}
                    />
                </mesh>
                
                {/* Eyes */}
                <mesh position={[-0.3, 0.3, 0.8]}>
                    <sphereGeometry args={[0.2, 32, 32]} />
                    <meshStandardMaterial color="#ffffff" />
                </mesh>
                <mesh position={[0.3, 0.3, 0.8]}>
                    <sphereGeometry args={[0.2, 32, 32]} />
                    <meshStandardMaterial color="#ffffff" />
                </mesh>
                
                {/* Pupils */}
                <mesh position={[-0.3, 0.3, 1]}>
                    <sphereGeometry args={[0.1, 32, 32]} />
                    <meshStandardMaterial color="#000000" />
                </mesh>
                <mesh position={[0.3, 0.3, 1]}>
                    <sphereGeometry args={[0.1, 32, 32]} />
                    <meshStandardMaterial color="#000000" />
                </mesh>
                
                {/* Smile */}
                <mesh position={[0, 0, 0.8]} rotation={[0, 0, Math.PI]}>
                    <torusGeometry args={[0.4, 0.1, 16, 32, Math.PI]} />
                    <meshStandardMaterial color="#000000" />
                </mesh>
            </group>
        </Float>
    );
}

const AIDashboard = () => {
    return (
        <Box 
            sx={{ 
                position: 'relative', 
                width: '100%', 
                height: '100vh', 
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                overflow: 'hidden',
                background: 'linear-gradient(135deg, #0a0a2a 0%, #1a1a3a 100%)',
                color: 'white'
            }}
        >
            <Canvas
                camera={{ position: [0, 0, 5], fov: 45 }}
                style={{ 
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    zIndex: 1
                }}
            >
                <StarBackground />
                <Suspense fallback={null}>
                    <AIAssistant />
                </Suspense>
            </Canvas>

            <Container 
                maxWidth="md" 
                sx={{ 
                    position: 'relative', 
                    zIndex: 10, 
                    textAlign: 'center',
                    mt: 10
                }}
            >
                <Stack spacing={3} alignItems="center">
                    <motion.div
                        initial={{ opacity: 0, y: 50 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <Typography 
                            variant="h3" 
                            sx={{ 
                                fontWeight: 'bold', 
                                color: '#4a6cf7',
                                textShadow: '0 0 10px rgba(74, 108, 247, 0.5)'
                            }}
                        >
                            AI Trading Assistant
                        </Typography>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 50 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.3 }}
                    >
                        <Typography 
                            variant="h6" 
                            sx={{ 
                                maxWidth: 600, 
                                textAlign: 'center',
                                color: '#a0b4ff',
                                mb: 3
                            }}
                        >
                            Empowering your trading decisions with advanced AI-driven market analysis and predictive insights
                        </Typography>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5, delay: 0.6 }}
                    >
                        <Grid container spacing={2} justifyContent="center">
                            {[
                                { 
                                    icon: '🎯', 
                                    title: '16 Specialized Models', 
                                    color: '#3a5fff' 
                                },
                                { 
                                    icon: '📊', 
                                    title: 'Real-time Analysis', 
                                    color: '#4a6cf7' 
                                },
                                { 
                                    icon: '🚀', 
                                    title: 'Smart Features', 
                                    color: '#5a7cff' 
                                }
                            ].map((feature, index) => (
                                <Grid item key={index}>
                                    <Paper 
                                        sx={{ 
                                            p: 2, 
                                            background: 'rgba(58, 95, 255, 0.1)',
                                            border: `2px solid ${feature.color}`,
                                            borderRadius: 2,
                                            transition: 'all 0.3s ease',
                                            '&:hover': {
                                                transform: 'scale(1.05)',
                                                background: 'rgba(58, 95, 255, 0.2)'
                                            }
                                        }}
                                    >
                                        <Typography 
                                            variant="body1" 
                                            sx={{ 
                                                display: 'flex', 
                                                alignItems: 'center',
                                                color: feature.color,
                                                fontWeight: 'bold'
                                            }}
                                        >
                                            {feature.icon} {feature.title}
                                        </Typography>
                                    </Paper>
                                </Grid>
                            ))}
                        </Grid>
                    </motion.div>
                </Stack>
            </Container>
        </Box>
    );
};

export default AIDashboard;
